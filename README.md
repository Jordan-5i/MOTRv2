> Forked from: https://github.com/megvii-research/MOTRv2

# Demo演示MOTRv2部署到爱芯AX650平台
请参考 https://github.com/megvii-research/MOTRv2 原始repo配置环境与相关实验数据（该demo使用DanceTrack数据集）

## 导出onnx时主要修改以下内容

- 增加 `MOTR_ONNX`类来支持onnx的导出行为
- 在计算deformable attention切换到 `ms_deform_attn_core_pytorch` 函数
- 把 `NestedTensor`类以及 `Instances` 类拆开以方便导出
- 预计算出图像位置编码
- QIMv2模块单独导出成一个onnx
- nonzero用where函数替代，以支持pulsar2的量化
- 模型中三个权重数据`position.weight, query_embed.weight, yolox_embed.weight`以npy的格式单独保存到本地，onnx输入需要用到它们。

1. 执行导出onnx的脚本：
    ```bash
    python tools/export_onnx.py 
    ```
    会在当前目录生成`motrv2-no-mask-position.onnx`和`qim.onnx`

    > PS：导出的onnx需要经过onnxsim优化，以更好的支持后续puslar2的量化。
2. 验证onnx正确性以及生成量化数据集，执行脚本：
    ```bash
    python tools/run_onnx.py 
    ```
    会在当前目录生成calib_data目录，其中`.tar`文件是量化所需要的量化数据集，以及onnx输入需要的3个权重文件`position.weight.npy, query_embed.weight.npy, yolox_embed.weight.npy`。calib_data目录结构如下：
    ```bash 
    calib_data/
    ├── motrv2
    │   └── data_motrv2.tar
    └── qim
        └── data_qim.tar
    ```
> 导出onnx注意事项：在`tools/export_onnx.py`中需要设置 `max_objs` 和 `max_tracks`两个参数，目的是为了在编译模型时固定第一维的大小，以避免动态维上板推理带来的麻烦。

## 使用pulsar2工具链量化onnx模型
> 量化工具使用方法参考：https://npu.pages-git-ext.axera-tech.com/pulsar2-docs/index.html

已提供必要的量化配置json文件, 位宽为8bit：
- `config_motrv2.json`
- `config_qim.json`

> 需要在2个json文件中修改字段`"calibration_dataset": "data_qim.tar"`，使其指向上一步生成的tar文件量化数据集 

模型编译命令如下：
```bash
# motrv2模块
pulsar2 build --input motrv2-no-mask-position-sim.onnx --config config_motrv2.json --output_dir output_motr/

# QIMv2模块
pulsar2 build --input qim-sim.onnx --config config_qim.json --output_dir output_qim
```
最终会在 `--output_dir` 目录中生成对应的2个能运行在AX650平台上的compiled.axmodel文件

## 在AX650平台上运行axmodel

把以下必要的文件上传到板子
- DanceTrack的数据集，本demo中只测试了 `DanceTrack/test/dancetrack0011/` 一个文件夹
- motrv2，qim编译后的2个axmodel
- `det_db_motrv2.json`
- `position.weight.npy, query_embed.weight.npy, yolox_embed.weight.npy` 3个axmodel输入需要用到的权重文件
- `models/structures/instances.py`，`tools/run_axmodel.py`

```bash
# 分别把2个compiled.axmodel重命名
scp output_motr/compiled.axmodel root@10.168.232.183:/root/wangjian/motrv2/motr-complied.axmodel

scp output_qim/compiled.axmodel root@10.168.232.183:/root/wangjian/motrv2/qim-complied.axmodel
```

板子上的目录结构如下：
```bash
/root/wangjian/motrv2
├── instances.py
├── DanceTrack
│   └── test
│       └── dancetrack0011
│           ├── seqinfo.ini
│           └── img1
├── det_db_motrv2.json
├── position.weight.npy
├── query_embed.weight.npy
├── yolox_embed.weight.npy
├── motr-complied.axmodel
├── qim-complied.axmodel
└── run_axmodel.py
```
执行脚本：
```bash
python run_axmodel.py
```
即可在当前目录中生成axmodel_output目录，板端运行结果保存在此目录中。

把板端结果scp会本地后，可视化推理结果，执行脚本：
```bash 
python tools/visualize.py
```
可视化结果如：https://github.com/Jordan-5i/MOTRv2/blob/main/dancetrack0011.mp4

