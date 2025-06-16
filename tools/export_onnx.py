from copy import deepcopy
import json
import sys

sys.path.append(".")

import os
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
import tarfile
import numpy as np
from tqdm import tqdm
from pathlib import Path
from models import build_model
from util.tool import load_model
from main import get_args_parser

from models.structures import Instances
from torch.utils.data import Dataset, DataLoader
from util.misc import nested_tensor_from_tensor_list, NestedTensor


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


class Detector(object):
    def __init__(self, args, model, vid):
        self.args = args
        self.detr = model

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

    @torch.no_grad()
    def prepare_mask_pos_encoding(self, samples):
        features, pos = self.detr.backbone(samples)
        src, mask = features[-1].decompose()

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.detr.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.detr.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1].tensors)
                else:
                    src = self.detr.input_proj[l](srcs[-1])
                m = samples.mask
                mask = torch.nn.functional.interpolate(
                    m[None].float(), size=src.shape[-2:]
                ).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        return masks, pos

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
        for i, data in enumerate(tqdm(loader)):
            cur_img, ori_img, proposals = [d[0] for d in data]
            cur_img, proposals = cur_img.cuda(), proposals.cuda()

            # track_instances = None
            if track_instances is not None:
                track_instances.remove("boxes")
                track_instances.remove("labels")
            seq_h, seq_w, _ = ori_img.shape

            track_instances_onnx = deepcopy(track_instances)
            
            ori_detr = self.detr.__class__.__bases__[0]
            res = ori_detr.inference_single_image(
                self.detr, cur_img, (seq_h, seq_w), track_instances, proposals
            )
            track_instances = res["track_instances"]

            if i > 20:
                cur_img, proposals = cur_img.cpu(), proposals.cpu()
                self.detr.cpu()
                if track_instances_onnx is None:
                    track_instances_onnx = self.detr._generate_empty_tracks(proposals)
                else:
                    track_instances_onnx = Instances.cat(
                        [
                            self.detr._generate_empty_tracks(proposals),
                            track_instances_onnx.to("cpu"),
                        ]
                    )
                query_pos = track_instances_onnx.query_pos.cpu()
                ref_pts = track_instances_onnx.ref_pts.cpu()
                samples = nested_tensor_from_tensor_list(cur_img)

                # 预计算位置编码，和masks中有效比例
                masks, pos = self.prepare_mask_pos_encoding(samples)
                valid_ratios = torch.stack(
                    [self.detr.transformer.get_valid_ratio(m) for m in masks], 1
                )

                setattr(self.detr.transformer, "valid_ratios", valid_ratios)
                setattr(self.detr, "pos", pos)
                setattr(self.detr, "masks", masks)

                # 保存onnx输入要用的weight
                np.save(
                    "position.weight.npy", self.detr.position.weight.detach().numpy()
                )
                np.save(
                    "query_embed.weight.npy",
                    self.detr.query_embed.weight.detach().numpy(),
                )
                np.save(
                    "yolox_embed.weight.npy",
                    self.detr.yolox_embed.weight.detach().numpy(),
                )

                self.detr.forward = self.detr.inference_single_image
                dynamic_axes = {
                    'query_pos': {0: 'num_queries'},  # 22 -> 动态
                    'ref_pts': {0: 'num_queries'}     # 22 -> 动态
                }
                torch.onnx.export(
                    self.detr,
                    (cur_img, query_pos, ref_pts),
                    "motrv2-no-mask-position-dynamic.onnx",
                    input_names=[
                        "cur_img",
                        "query_pos",
                        "ref_pts",
                    ],
                    dynamic_axes=dynamic_axes,
                    opset_version=16,
                )

                # ----- qim -----#
                track_instances = track_instances[track_instances.obj_idxes >= 0]

                query_pos = track_instances.query_pos.cpu()
                ref_pts = track_instances.ref_pts.cpu()
                scores = track_instances.scores.cpu()
                output_embedding = track_instances.output_embedding.cpu()
                pred_boxes = track_instances.pred_boxes.cpu()

                self.detr.track_embed.forward = self.detr.track_embed.onnx_forward
                dynamic_axes = {
                    'pred_boxes': {0: 'num_queries'},
                    'output_embedding': {0: 'num_queries'},
                    'query_pos': {0: 'num_queries'},  # 22 -> 动态
                    'ref_pts': {0: 'num_queries'},     # 22 -> 动态
                    'scores': {0: 'num_queries'},
                }
                torch.onnx.export(
                    self.detr.track_embed,
                    (pred_boxes, output_embedding, ref_pts, query_pos, scores),
                    "qim-dynamic.onnx",
                    input_names=[
                        "pred_boxes",
                        "output_embedding",
                        "ref_pts",
                        "query_pos",
                        "scores",
                    ],
                    output_names=["output_embedding_", "query_pos_"],
                    dynamic_axes=dynamic_axes,
                    opset_version=16,
                )

                sys.exit()

            dt_instances = deepcopy(track_instances)

            # filter det instances by score.
            dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
            dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

            total_dts += len(dt_instances)

            bbox_xyxy = dt_instances.boxes.tolist()
            identities = dt_instances.obj_idxes.tolist()

            save_format = "{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
            for xyxy, track_id in zip(bbox_xyxy, identities):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                lines.append(
                    save_format.format(frame=i + 1, id=track_id, x1=x1, y1=y1, w=w, h=h)
                )
        with open(os.path.join(self.predict_path, f"{self.seq_num}.txt"), "w") as f:
            f.writelines(lines)
        print("totally {} dts {} occlusion dts".format(total_dts, total_occlusion_dts))


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    parser.add_argument("--score_threshold", default=0.5, type=float)
    parser.add_argument("--update_score_threshold", default=0.5, type=float)
    parser.add_argument("--miss_tolerance", default=20, type=int)
    parser.add_argument(
        "--onnx_path", default="motrv2.onnx", type=str, help="onnx path"
    )
    args = parser.parse_args()

    args.meta_arch = "motr"
    args.resume = "pretrained/motrv2_dancetrack.pth"
    args.mot_path = "/data/wangjian/project/hf_cache"
    args.det_db = "/data/wangjian/project/hf_cache/DanceTrack/det_db_motrv2.json"
    args.dataset_file = "e2e_dance"
    args.pretrained = "/data/wangjian/project/MOTRv2/pretrained/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth"
    args.query_interaction_layer = "QIMv2"
    args.use_checkpoint = True
    args.sample_mode = "random_interval"
    args.sampler_lengths = [5]
    args.sample_interval = 10
    args.merger_dropout = 0
    args.dropout = 0
    args.random_drop = 0.1
    args.fp_ratio = 0.3
    args.query_denoise = 0.05
    args.num_queries = 10
    args.append_crowd = True
    args.with_box_refine = True
    args.batch_size = 1
    args.output_dir = "."
    args.export = True
    
    # load model and weights
    detr, _, _ = build_model(args)
    detr.track_embed.score_thr = args.update_score_threshold
    detr.track_base = RuntimeTrackerBase(
        args.score_threshold, args.score_threshold, args.miss_tolerance
    )
    checkpoint = torch.load(args.resume, map_location="cpu")
    detr = load_model(detr, args.resume)
    detr.eval()
    detr = detr.cuda()

    # '''for MOT17 submit'''
    sub_dir = "DanceTrack/test"
    # seq_nums = os.listdir(os.path.join(args.mot_path, sub_dir))[:1]
    seq_nums = ["dancetrack0064"]
    if "seqmap" in seq_nums:
        seq_nums.remove("seqmap")
    vids = [os.path.join(sub_dir, seq) for seq in seq_nums]

    rank = int(os.environ.get("RLAUNCH_REPLICA", "0"))
    ws = int(os.environ.get("RLAUNCH_REPLICA_TOTAL", "1"))
    vids = vids[rank::ws]

    for vid in vids:
        det = Detector(args, model=detr, vid=vid)
        det.detect(args.score_threshold)
