# HGSFusion 自复现基线与后续实验闸门（2026-10-06）

## 已核实的数据和运行条件

运行 `python tools/research/audit_data.py --output /tmp/hgsfusion_data_audit.json` 可重复文件计数与逐帧缺失检查；脚本仅读取数据。VoD 雷达与独立 LiDAR 均有 5139/1296 个 train/val ID，分别是 `Nx7` 与 `Nx4`；相应雷达、LiDAR、相机、标定和预生成点在 train/val 均无缺失。TJ4DRadSet 有 5706/2040 个 train/val ID，`velodyne` 是 `Nx8` **雷达**；目录中没有独立 LiDAR 或 LiDAR 到雷达的标定。`velodyne3` 和 `velodyne_reduced` 不能仅凭目录名当 LiDAR。原始数据挂载位于 `/mnt/data2`，脚本不写该目录。

当前执行容器的 Python 无 NumPy，`nvidia-smi` 无法连接驱动。因此本次无法读取 `.pkl.npy` 内容、运行模型或绘制实际叠图；以下数值来自已经存在的评测日志，不是本次新跑的实验。执行训练时须先进入原复现环境并确认 `python -c 'import numpy,torch; print(torch.cuda.is_available())'` 为 True。

## 固定自复现对照

| 数据集 | 权重与口径 | 自复现结果 | 训练设置 |
| --- | --- | --- | --- |
| VoD val，1296 帧 | epoch 25；Entire / Driving Corridor 3D mAP | 57.416 / 76.413 | 5 进程，每卡 batch 1，有效 batch 5，AMP，sync BN |
| TJ4D val，2040 帧 | epoch 31；all_weather BEV / 3D AP40，moderate | 28.888 / 25.417 | 4 进程，每卡 batch 1，有效 batch 4，FP32，无 sync BN |

VoD epoch 24 的 RoI 3D mAP 为 76.580，但 EAA 3D mAP 为 57.322；必须预先规定主指标，避免按不同指标挑不同权重。以上固定 VoD epoch 25（EAA 最优）和 TJ4D epoch 31（已评估的 31–40 中 all_weather BEV moderate 最优）。仓库内作者权重按本地日志分别得到 VoD EAA/RoI 58.965/79.460，TJ4D all_weather BEV/3D moderate 43.266/37.228。TJ4D 自训与作者权重差距显著，不能称为“约 3 点”；原因尚未确定。TJ4D 训练日志的 sync BN 为 False，和项目 README 的复现提示不符，应记为差异，给复现实验设定有限的核查时间后继续创新验证。两个训练日志均未固定随机种子。训练时保存的配置与当前配置 SHA256 一致，但未记录数据文件与代码提交的哈希；本报告固定的是**当前仓库和现有权重**，后续须另存 Git diff 与数据清单。

评测入口：`tools/test.py` → `tools/eval_utils/eval_utils.py` → `VODDataset.evaluation`/`TJ4DDataset.evaluation`。VoD 使用 `pcdet/utils/vod_utils.py` 的 EAA 和 Driving Corridor 3D AP；TJ4D 使用 `pcdet/datasets/kitti/tj4d_utils.py` 的 AP40 和天气子集。TJ4D 的 `val.txt` 与 `test.txt` ID 相同，当前实验必须标为 **val**，不能当盲测集。

## 第一阶段：实际代码链路与张量

`hybrid_pts/hybrid_radar_pts_{vod,tj4d}.py` 在离线 RHGM 中用 MaskFormer 掩码与真实雷达点生成虚拟点，结果存为 `.pkl.npy`。其 `process_one_frame` 读取的是 radar `velodyne`，未在该函数读取 3D 真值框或 LiDAR；仍需对现存预生成文件的来源做哈希和样本核验。`{vod,tj4d}_dataset.py::__getitem__` 拼接原始雷达、掩码前景雷达与虚拟点，随后 FOV 过滤、训练增强；检测标签则从 `annos` 单独取得。现有推理依赖预生成点文件，未来扩散推理必须替换该读取路径并证明文件仅由雷达/相机生成。

坐标系：`velodyne` 点与 `gt_boxes_lidar` 在当前数据集的传感器坐标系；在 VoD 雷达数据集里该所谓 `lidar` 坐标实际是**雷达坐标**。`Calibration.V2C` 是该坐标到相机坐标的变换，`P2` 做投影。VoD 教师 LiDAR 到雷达应为 `inv(T_camera_from_radar) @ T_camera_from_lidar`，并处理 `R0_rect`。增强后的 BEV 坐标由 `lidar_aug_matrix` 给出；生成器与检测器必须在同一增强坐标下比较。

VoD 原始点 `N×7`，预生成前景/虚拟点 `N×15`，拼接后 `N×17`（末两位是点类型）；TJ4D 分别为 `N×8`、`N×16`、`N×18`。批处理点张量再前置 batch 索引。`DynamicPillarFeatureNet` 以点类型把原始特征和虚拟特征分开，VoD/TJ4D 分别输入 29/31 维。图像 `ImageVFE` 经多尺度 `FrustumToVoxelMultiScale` 投到 BEV；VoD 配置体素 XY 为 320×320，TJ4D 为 216×248。`RadarOccupancy2DV2` 将雷达占用映射用于图像 BEV，`FusionAfterBEVSEDirect` 合并图像和雷达的各 128 通道为 256 通道，经卷积和 SE 后进入 `AnchorHeadSingle`。这里的空间同步由占用图实现，模态同步由 BEV 融合实现；不能把 RHGM 当作模型内可微层。

## 第二阶段：样本制作准入

只在 VoD train 读取独立 LiDAR `Nx4`。先按上述外参变换到雷达坐标，再按 VoD 范围 `[0,-25.6,-3,51.2,25.6,2]`、正相机深度和图像边界过滤。相机前景应来自**推理可用的 MaskFormer 输出**，雷达候选区域来自雷达点及其物理合理的距离/方位邻域。LiDAR 仅提供 xyz 几何教师目标；目标文件不能含 LiDAR RCS、径向速度、真值框或验证集监督。需逐帧生成相机平面和 BEV 两张叠图，展示雷达、LiDAR 教师、掩码和筛选后的点，并计算投影落入掩码率和最近雷达距离分布。现有 `.pkl.npy` 只存点，没有原始像素掩码；必须找到/重新生成掩码后才能严格完成此阶段。TJ4D 无可用 LiDAR 时做无 LiDAR 迁移/测试，不能声称 TJ4D 上也做了 LiDAR 教师训练。

## 后续实验闸门

1. 固定上述两条自训基线及评测脚本；TJ4D 最多做一次对齐的 sync BN/有效 batch 核查，保留差异记录，随后推进 VoD。重新运行命令见下方。
2. 在 VoD train 制作监督样本、可视化 20 帧并人工核对；不合格就修正坐标与掩码，不进入扩散训练。
3. 用小型条件扩散生成 xyz，原 HGSFusion 权重和其他模块固定，推理禁用原虚拟点读取；与原 RHGM、无虚拟点、简单随机/高斯补点比较同口径 AP 与整条流水线延迟。若无稳定增益，先缩小生成目标和候选区域。
4. 仅在第三步收益成立后，输出生成点几何置信度；按 BEV 位置汇聚置信度，对生成分支门控，保留原始雷达分支。做 Detection Loss 联合微调，并核查高置信生成点的几何误差确实更小。
5. 四格消融：原 HGSFusion、仅扩散、仅新融合、两者同时；同训练预算和种子，另测采样步数/延迟。TJ4D 用自身 AP40/天气划分，VoD 用 EAA/RoI 3D AP，不跨集平均。若效果不稳，论文收缩为受约束的几何生成和误差分析。

第二篇独立选题限定在相机退化/夜间条件下的**模态可靠性**：用亮度、模糊或遮挡控制相机质量，评测不同退化强度和 TJ4D 天气子集；不共享第一篇的生成置信度门控作为创新点。

## 先前工作与投稿窗口

WACV 2024 的 [Wu 等](https://openaccess.thecvf.com/content/WACV2024/html/Wu_A_Robust_Diffusion_Modeling_Framework_for_Radar_Camera_3D_Object_WACV_2024_paper.html) 已做雷达相机扩散检测；[RCTDistill](https://openaccess.thecvf.com/content/ICCV2025/html/Bang_RCTDistill_Cross-Modal_Knowledge_Distillation_Framework_for_Radar-Camera_3D_Object_Detection_ICCV_2025_paper.html) 已做训练期 LiDAR 教师与雷达相机 BEV 蒸馏。第一篇必须把区别落在**受 LiDAR 几何监督的虚拟 xyz 点生成及其几何不确定性向融合传递**，不能写“首次扩散”或“首次 LiDAR 教师”。

截至本报告日期，[IEEE IPAS 2027](https://events.vtools.ieee.org/m/563249) 列 2026-11-10 截稿、2026-12-15 通知；[IEEE IV 2027 页面](https://ieee-iv.org/2027/intelligent-vehicles-symposium-2027-perth-australia/) 列 2026-11-15 截稿、2027-01-15 通知；[IEEE MOST 2027](https://www.ieeemobility.org/) 列 2026-12-01 截稿、2027-01-15 通知。IV 的另一篇[官方旧公告](https://ieee-iv.org/2026/blog/category/announcements/)曾称 2027 起恢复 10 月 1 日固定截止，与目前 2027 页面冲突，需以最终 CFP/投稿系统为准。以上均为**2026 年**截稿，2027 是会议年份。EI Compendex 的具体届次收录、正式论文集及学校是否认可录用通知尚未核实，投稿前必须分别拿到证据。

## 可运行命令与文件

```bash
python tools/research/audit_data.py --output /tmp/hgsfusion_data_audit.json
sha256sum tools/cfgs/hgsfusion/hgsfusion_{vod,tj4d}.yaml
git diff > /tmp/hgsfusion_baseline_worktree.diff

# 在可用 GPU 的原训练环境中，从仓库根目录按固定 checkpoint 重新评测。
python tools/test.py --cfg_file tools/cfgs/hgsfusion/hgsfusion_vod.yaml --batch_size 1 --ckpt output/cfgs/hgsfusion/hgsfusion_vod/vod_hgsfusion_4gpu/ckpt/checkpoint_epoch_25.pth --extra_tag fixed_self_vod
python tools/test.py --cfg_file tools/cfgs/hgsfusion/hgsfusion_tj4d.yaml --batch_size 1 --ckpt output/cfgs/hgsfusion/hgsfusion_tj4d/tj4d_4GPU_FP32/ckpt/checkpoint_epoch_31.pth --extra_tag fixed_self_tj4d
```

本次新增 `tools/research/audit_data.py` 和本文档；未改动已有模型或配置。当前 GPU/NumPy 不可用，第二阶段的实际叠图、生成训练、消融和耗时尚无结果，不应填写推测数字。
