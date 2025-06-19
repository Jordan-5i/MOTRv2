from copy import deepcopy
import json
import sys

sys.path.append(".")

import os
import math
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
import tarfile
import numpy as np
import onnxruntime
from torch import nn
from tqdm import tqdm
from pathlib import Path
from models import build_model
from util.tool import load_model
from main import get_args_parser

from models.structures import Instances
from torch.utils.data import Dataset, DataLoader
from util.misc import nested_tensor_from_tensor_list


def pos2posemb(pos, num_pos_feats=64, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    posemb = pos[..., None] / dim_t
    posemb = torch.stack(
        (posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1
    ).flatten(-3)
    return posemb


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


class ListImgDataset(Dataset):
    def __init__(self, mot_path, img_list, det_db) -> None:
        super().__init__()
        self.mot_path = mot_path
        self.img_list = img_list
        self.det_db = det_db

        """
        common settings
        """
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        cur_img = cv2.imread(os.path.join(self.mot_path, f_path))
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        proposals = []
        im_h, im_w = cur_img.shape[:2]
        for line in self.det_db[f_path[:-4] + ".txt"]:
            l, t, w, h, s = list(map(float, line.split(",")))
            proposals.append(
                [(l + w / 2) / im_w, (t + h / 2) / im_h, w / im_w, h / im_h, s]
            )
        return cur_img, torch.as_tensor(proposals).reshape(-1, 5)

    def init_img(self, img, proposals):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img, proposals

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, proposals = self.load_img_from_file(self.img_list[index])
        return self.init_img(img, proposals)


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_obj = (track_instances.obj_idxes == -1) & (
            track_instances.scores >= self.score_thresh
        )
        disappeared_obj = (track_instances.obj_idxes >= 0) & (
            track_instances.scores < self.filter_score_thresh
        )
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(
            num_new_objs, device=device
        )
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (
            track_instances.disappear_time >= self.miss_tolerance
        )
        track_instances.obj_idxes[to_del] = -1


class TrackerPostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes

        # prob = out_logits.sigmoid()
        scores = out_logits[..., 0].sigmoid()
        # scores, labels = prob.max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = torch.full_like(scores, 0)
        # track_instances.remove('pred_logits')
        # track_instances.remove('pred_boxes')
        return track_instances


class Detector(object):
    def __init__(self, args, model, track_embed, vid):
        self.args = args
        self.num_classes = 1
        self.detr = model
        self.track_embed = track_embed  # QIMv2 模块
        self.track_base = RuntimeTrackerBase(args.score_threshold, args.score_threshold, args.miss_tolerance) # 对应submit_dance.py中args设置的默认值
        self.post_process = TrackerPostProcess()  # 把输出结果还原回原图大小
        self.memory_bank = None
        self.query_denoise = 0.05
        
        os.makedirs("calib_data/motrv2", exist_ok=True)
        os.makedirs("calib_data/qim", exist_ok=True)
        self.calib_tarfile_motr = tarfile.open(
            f"calib_data/motrv2/data_motrv2.tar", "w"
        )
        self.calib_tarfile_qim = tarfile.open(f"calib_data/qim/data_qim.tar", "w")

        self.vid = vid
        self.seq_num = os.path.basename(vid)
        img_list = os.listdir(os.path.join(self.args.mot_path, vid, "img1"))
        img_list = [os.path.join(vid, "img1", i) for i in img_list if "jpg" in i]

        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)

        self.predict_path = os.path.join(self.args.output_dir, args.exp_name)
        os.makedirs(self.predict_path, exist_ok=True)

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    def _fill_tracks_to_max_len(self, track_instances: Instances, max_objs=50):
        d_model = track_instances.query_pos.shape[-1]
        device = track_instances.query_pos.device
        remain_len = max_objs - len(track_instances)
        
        placeholder = Instances((1, 1))
        placeholder.ref_pts = torch.zeros((remain_len, 4))
        placeholder.query_pos = torch.zeros((remain_len, d_model))
        
        placeholder.output_embedding = torch.zeros(
            (remain_len, d_model), device=device
        )
        placeholder.obj_idxes = torch.full(
            (remain_len,), -1, dtype=torch.long, device=device
        )
        placeholder.matched_gt_idxes = torch.full(
            (remain_len,), -1, dtype=torch.long, device=device
        )
        placeholder.disappear_time = torch.zeros(
            (remain_len,), dtype=torch.long, device=device
        )
        placeholder.iou = torch.ones(
            (remain_len,), dtype=torch.float, device=device
        )
        placeholder.scores = torch.zeros(
            (remain_len,), dtype=torch.float, device=device
        )
        placeholder.track_scores = torch.zeros(
            (remain_len,), dtype=torch.float, device=device
        )
        placeholder.pred_boxes = torch.zeros(
            (remain_len, 4), dtype=torch.float, device=device
        )
        placeholder.pred_logits = torch.zeros(
            (remain_len, self.num_classes), dtype=torch.float, device=device
        )

        mem_bank_len = 0
        placeholder.mem_bank = torch.zeros(
            (remain_len, mem_bank_len, d_model),
            dtype=torch.float32,
            device=device,
        )
        placeholder.mem_padding_mask = torch.ones(
            (remain_len, mem_bank_len), dtype=torch.bool, device=device
        )
        placeholder.save_period = torch.zeros(
            (remain_len,), dtype=torch.float32, device=device
        )
        return Instances.cat([track_instances, placeholder])
    
    def _generate_empty_tracks(self, proposals=None):
        query_embed_weight = torch.from_numpy(np.load("query_embed.weight.npy"))
        yolox_embed_weight = torch.from_numpy(np.load("yolox_embed.weight.npy"))
        position_weight = torch.from_numpy(np.load("position.weight.npy"))
        track_instances = Instances((1, 1))
        num_queries, d_model = query_embed_weight.shape  # (300, 512)
        device = query_embed_weight.device
        if proposals is None:
            track_instances.ref_pts = position_weight
            track_instances.query_pos = query_embed_weight
        else:
            track_instances.ref_pts = torch.cat([position_weight, proposals[:, :4]])
            track_instances.query_pos = torch.cat(
                [
                    query_embed_weight,
                    pos2posemb(proposals[:, 4:], d_model) + yolox_embed_weight,
                ]
            )
        track_instances.output_embedding = torch.zeros(
            (len(track_instances), d_model), device=device
        )
        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.disappear_time = torch.zeros(
            (len(track_instances),), dtype=torch.long, device=device
        )
        track_instances.iou = torch.ones(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 4), dtype=torch.float, device=device
        )
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        mem_bank_len = 0
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, d_model),
            dtype=torch.float32,
            device=device,
        )
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len), dtype=torch.bool, device=device
        )
        track_instances.save_period = torch.zeros(
            (len(track_instances),), dtype=torch.float32, device=device
        )

        return track_instances

    def _post_process_single_image(self, frame_res, track_instances, is_last, frame_id):
        if self.query_denoise > 0:
            n_ins = len(track_instances)
            ps_logits = frame_res['pred_logits'][:, n_ins:]
            ps_boxes = frame_res['pred_boxes'][:, n_ins:]
            frame_res['hs'] = frame_res['hs'][:, :n_ins]
            frame_res['pred_logits'] = frame_res['pred_logits'][:, :n_ins]
            frame_res['pred_boxes'] = frame_res['pred_boxes'][:, :n_ins]
            ps_outputs = [{'pred_logits': ps_logits, 'pred_boxes': ps_boxes}]
            for aux_outputs in frame_res['aux_outputs']:
                ps_outputs.append({
                    'pred_logits': aux_outputs['pred_logits'][:, n_ins:],
                    'pred_boxes': aux_outputs['pred_boxes'][:, n_ins:],
                })
                aux_outputs['pred_logits'] = aux_outputs['pred_logits'][:, :n_ins]
                aux_outputs['pred_boxes'] = aux_outputs['pred_boxes'][:, :n_ins]
            frame_res['ps_outputs'] = ps_outputs
            
        track_scores = frame_res["pred_logits"][0, :, 0].sigmoid()
        print(track_scores)
        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res["pred_logits"][0]
        track_instances.pred_boxes = frame_res["pred_boxes"][0]
        track_instances.output_embedding = frame_res["hs"][0]

        # each track will be assigned an unique global id by the track base.
        self.track_base.update(track_instances)
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        if not is_last:
            # 对应 QIM._select_active_tracks()
            track_instances = track_instances[track_instances.obj_idxes >= 0]
            
            # 填充无效数据将第一维大小变成args.max_tracks
            track_instances = self._fill_tracks_to_max_len(track_instances, args.max_tracks)
            
            # track_embed 对应 QIM._update_track_embedding()
            inputs = {
                "pred_boxes": track_instances.pred_boxes.numpy(),
                "output_embedding": track_instances.output_embedding.numpy(),
                "ref_pts": track_instances.ref_pts.numpy(),
                "query_pos": track_instances.query_pos.numpy(),
                "scores": track_instances.scores.numpy(),
            }

            # -------- 保存量化校准数据 ------------- #
            if frame_id < 20:
                np.save(f"calib_data/qim/data_qim_{frame_id}.npy", inputs)
                self.calib_tarfile_qim.add(f"calib_data/qim/data_qim_{frame_id}.npy")
            # -------------------------------------- #

            ref_pts, query_pos = self.track_embed.run(None, inputs)

            track_instances.query_pos = torch.from_numpy(query_pos)
            track_instances.ref_pts = torch.from_numpy(ref_pts)
            frame_res["track_instances"] = track_instances
        else:
            frame_res["track_instances"] = None
        return frame_res

    @torch.no_grad()
    def detect(self, prob_threshold=0.6, area_threshold=100, vis=False):
        total_dts = 0
        total_occlusion_dts = 0

        track_instances = None
        with open(os.path.join(self.args.mot_path, self.args.det_db)) as f:
            det_db = json.load(f)
        loader = DataLoader(
            ListImgDataset(self.args.mot_path, self.img_list, det_db), 1, num_workers=2
        )
        lines = []
        try:
            for i, data in enumerate(tqdm(loader)):
                cur_img, ori_img, proposals = [d[0] for d in data]

                # track_instances = None
                if track_instances is not None:
                    track_instances.remove("boxes")
                    track_instances.remove("labels")
                seq_h, seq_w, _ = ori_img.shape
                ori_img_size = (seq_h, seq_w)

                if track_instances is None:
                    track_instances = self._generate_empty_tracks(proposals)
                    track_instances = self._fill_tracks_to_max_len(track_instances, args.max_objs)
                else:
                    track_instances = Instances.cat(
                        [self._generate_empty_tracks(proposals), track_instances]
                    )
                    track_instances = self._fill_tracks_to_max_len(track_instances, args.max_objs)

                inputs = {
                    "cur_img": cur_img.numpy(),
                    "query_pos": track_instances.query_pos.numpy(),
                    "ref_pts": track_instances.ref_pts.numpy(),
                }

                res = self.detr.run(None, inputs)

                frame_res = {
                    "pred_logits": torch.from_numpy(res[0]),  # (1, 22, 1), 只有人一个类别，第6个layer的输出
                    "pred_boxes": torch.from_numpy(res[1]),  # (1, 22, 4)
                    "hs": torch.from_numpy(res[-1]),  # (1, 22, 256)
                    "aux_outputs": [{'pred_logits': torch.from_numpy(res[i]), 'pred_boxes': torch.from_numpy(res[i+1])} for i in [2, 4, 6, 8, 10]] # 第1~5个layer的输出
                }
                
                # -----------保存量化校准数据------------ #
                if i < 20:
                    np.save(f"calib_data/motrv2/data_motrv2_{i}.npy", inputs)
                    self.calib_tarfile_motr.add(
                        f"calib_data/motrv2/data_motrv2_{i}.npy"
                    )
                # ------------------------------------- #

                # 对应 MOTR._post_process_single_image()
                is_last = False
                frame_res = self._post_process_single_image(
                    frame_res, track_instances, is_last, i
                )

                track_instances = frame_res["track_instances"]
                track_instances = self.post_process(track_instances, ori_img_size)
                ret = {"track_instances": track_instances}
                if "ref_pts" in frame_res:
                    ref_pts = frame_res["ref_pts"]
                    img_h, img_w = ori_img_size
                    scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
                    ref_pts = ref_pts * scale_fct[None]
                    ret["ref_pts"] = ref_pts
                track_instances = ret["track_instances"]

                dt_instances = deepcopy(track_instances)

                # filter det instances by score.
                dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
                dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

                total_dts += len(dt_instances)

                bbox_xyxy = dt_instances.boxes.tolist()
                identities = dt_instances.obj_idxes.tolist()

                save_format = (
                    "{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
                )
                for xyxy, track_id in zip(bbox_xyxy, identities):
                    if track_id < 0 or track_id is None:
                        continue
                    x1, y1, x2, y2 = xyxy
                    w, h = x2 - x1, y2 - y1
                    lines.append(
                        save_format.format(
                            frame=i + 1, id=track_id, x1=x1, y1=y1, w=w, h=h
                        )
                    )

            self.calib_tarfile_motr.close()
            self.calib_tarfile_qim.close()

            with open(os.path.join(self.predict_path, f"{self.seq_num}.txt"), "w") as f:
                f.writelines(lines)
            print(
                "totally {} dts {} occlusion dts".format(total_dts, total_occlusion_dts)
            )
        except (Exception, KeyboardInterrupt) as e:
            print(e)
            self.calib_tarfile_motr.close()
            self.calib_tarfile_qim.close()

            with open(os.path.join(self.predict_path, f"{self.seq_num}.txt"), "w") as f:
                f.writelines(lines)
            print(
                "totally {} dts {} occlusion dts".format(total_dts, total_occlusion_dts)
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser("run axmodel")
    parser.add_argument("--score_threshold", default=0.5, type=float)
    parser.add_argument("--update_score_threshold", default=0.5, type=float)
    parser.add_argument("--miss_tolerance", default=20, type=int)
    args = parser.parse_args()

    args.output_dir = "."
    args.mot_path = "/data/wangjian/project/hf_cache"
    args.exp_name = "onnx_output"
    args.det_db = "/data/wangjian/project/hf_cache/DanceTrack/det_db_motrv2.json"

    # max_objs=(object_query+proposals+track_query)+填充的无效tensor, 目的为了消除decoder的输入query_pos，ref_pts中第一维是动态维，设置此参数.
    args.max_objs = 50
    # max_tracks是用于QIMv2模块，定义QIM中能够跟踪到的最大目标数
    args.max_tracks = 10
    
    motr_model_path = "motrv2-max-50-obj-sim.onnx"
    qim_model_path = "qim-sim.onnx"
    motr_model = onnxruntime.InferenceSession(motr_model_path)
    qim_model = onnxruntime.InferenceSession(qim_model_path)

    # '''for MOT17 submit'''
    sub_dir = "DanceTrack/test"
    # seq_nums = os.listdir(os.path.join(args.mot_path, sub_dir))[:1]
    seq_nums = ["dancetrack0011"]
    if "seqmap" in seq_nums:
        seq_nums.remove("seqmap")
    vids = [os.path.join(sub_dir, seq) for seq in seq_nums]

    for vid in vids:
        det = Detector(args, model=motr_model, track_embed=qim_model, vid=vid)
        det.detect(args.score_threshold)
