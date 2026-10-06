# HGSFusion 权重在挂载盘上的位置

整理时间：2026-10-06。权重集中于 `/mnt/data2/zhangyan/hgsfusion_weights`，包含 `pretrained/`、`official/` 和 `runs/hgsfusion_{vod,tj4d}/<实验标签>/`。`manifest.json` 记录了 62 条旧路径、目标路径、字节数和 SHA256；两份相同的 DeepLabV3 预训练权重共用一个目标文件，因此挂载盘有 61 个唯一权重文件，约 17.44 GB。复制时逐文件计算源与目标 SHA256，一致后才替换旧路径。

仓库根目录的 `hgsfusion_vod.pth`、`hgsfusion_tj4d.pth`、`checkpoints/*.pth`，以及上一级 `../checkpoints/*.pth` 均保留为符号链接。已有实验的 `output/.../ckpt` 目录整体指向挂载盘，原训练和评测命令仍能沿旧路径加载；在这些实验上续训产生的新 checkpoint 也会写入挂载盘。日志、评测结果和 TensorBoard 仍位于 `output/`。

新实验开始前先建立对应的 ckpt 目录链接，再运行训练：

```bash
python tools/research/prepare_ckpt_dir.py --dataset hgsfusion_vod --tag <实验标签>
# TJ4D 用 --dataset hgsfusion_tj4d
```

`<实验标签>` 必须与 `tools/train.py --extra_tag` 完全一致。迁移脚本 `python tools/research/migrate_weights.py` 默认仅列计划；`--mode link` 会复制、验证并链接发现的权重。挂载盘清单是权重哈希的记录，Git 仅跟踪迁移脚本和本文档，不跟踪模型文件。若挂载盘不可用，符号链接指向的权重也无法加载。
