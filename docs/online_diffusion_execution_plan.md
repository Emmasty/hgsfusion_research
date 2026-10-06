# 在线生成与两篇论文执行计划

日期：2026-10-06。工作仓库当前在 `main`，含 8 个已修改的复现文件、未跟踪的 `requirement.txt`/`requirements.txt`/`tj4d_pts.tar.gz`，以及新写的基线审计文档与脚本。仓库的 `origin` 是个人研究仓库，`hgsfusion` 是上游作者仓库。以下命令从仓库根目录运行；不要用 `git add .` 把压缩包和未经审阅的环境文件提交进去。

## 0. 保存并冻结现状（10 月 6–9 日）

```bash
git status --short
git diff > /tmp/hgsfusion_prebaseline_20261006.patch
git switch -c research/baseline-freeze
git diff -- pcdet/models tools/cfgs tools/test.py
python tools/research/audit_data.py --output /tmp/hgsfusion_data_audit.json
```

逐个审阅 8 个已改文件是否就是复现所需修复，单独审阅两个 requirements 文件；`tj4d_pts.tar.gz` 不入库。确认后用 `git add -p` 分批提交，先提交复现代码与配置，再提交 `tools/research/audit_data.py` 和 `docs/research_baseline_2026-10-06.md`。每次提交前运行 `git diff --cached --check` 和 `git diff --cached --stat`。不要重置或覆盖当前未提交内容。复现提交完成后，才用 `git tag -a baseline-hgsfusion-20261006 -m 'Fixed self-reproduction baseline'` 固定代码版本。Git 标签本身不包含权重，实验记录需另列权重 SHA256。

使用已存在的 VoD epoch 25、TJ4D epoch 31 权重为固定自复现对照；详情和复测命令见 `docs/research_baseline_2026-10-06.md`。记录：Git commit、配置与 checkpoint 哈希、数据清单、GPU/驱动/依赖版本、种子、实际卡数、每卡 batch、梯度累积、AMP、sync BN、完整评测命令及指标 JSON。`effective_batch = 每卡 batch × GPU 进程数 × 梯度累积步数`。当前环境无可用 GPU 和 NumPy，复测在原训练环境执行。TJ4D 自训与作者权重差距大，限定一次可比较的设置核查后继续实验。

## 1. VoD 几何教师和在线前景（10 月 9–15 日）

从 `research/baseline-freeze` 创建 `research/online-geometry`。数据集只读取原始雷达 `Nx7`、相机、标定和标签；训练分支另外读取独立 LiDAR `Nx4`，验证/推理分支不得读取 LiDAR、真值框或 `.pkl.npy` 虚拟点。可把训练期 LiDAR xyz 教师目标预计算以节省 I/O；**虚拟点本身必须在模型 forward 中生成**。教师点经 `T_radar_from_lidar = inv(T_camera_from_radar) @ T_camera_from_lidar` 变换，再限制点云范围、相机 FOV、前景掩码和雷达候选邻域。先让前景模型在线运行并冻结参数，稳定后才考虑解冻；输入到最终检测输出的推理流程始终完整在线。

建议文件：`pcdet/datasets/kitti/vod_dataset.py` 增加互斥的在线模式；`pcdet/models/backbones_3d/vfe/online_point_generator.py` 实现雷达点和图像/掩码条件编码；`tools/research/visualize_teacher.py` 绘制相机和 BEV 对齐图。每个训练样本输出 `radar_xyz, image, mask, teacher_xyz`，教师数据单独字段，不能拼进 detector 的 `points`。随机抽 20 帧检查投影叠图、掩码命中率和教师点到雷达距离分布；检查不通过就修坐标。

最小提交：`data: add training-only VoD geometry teacher and alignment audit`。提交代码、配置和少量去标识的可视化图；不提交原始帧、完整点云或预计算教师缓存。

## 2. 最小在线生成实验（10 月 16–25 日）

条件输入用雷达候选点、相机局部特征、在线掩码。先按雷达种子组织固定上限的候选槽，扩散只预测相对于雷达候选的 xyz 偏移；限制范围和点数。先训练小型去噪网络，以 `L_diff + λ L_geom` 学习 VoD 教师 xyz，优先试 2/4/8 步确定性采样。将其放入 `CaDDN.forward` 的 radar backbone 前，保留原始雷达点，并在推理时彻底绕开 `.pkl.npy`。先固定 HGSFusion 权重，只训练生成器并记录几何误差与整条推理链耗时；随后尝试检测损失联合训练。

当前 `pcdet/ops/pillar_ops/pillar_utils.py` 的 `group_point_features.detach()` 会阻断检测梯度。接入联合训练时要让生成点的**连续特征路径**保持梯度，并做一次反向传播断言：`Detection Loss` 对生成器参数的梯度有限且非零。柱索引是离散的，不能声称坐标到柱索引完全可微；生成 xyz 仍可经点特征和几何损失优化，必要时再增加软 BEV 投影。先避免把来自 LiDAR 的 RCS/速度写进生成点；物理雷达通道仅来自真实雷达测量，其余用独立的生成点编码。

同预算比较：固定原 RHGM、无虚拟点、随机/高斯补点、在线扩散。所有方法使用同一自训基线、VoD val EAA 与 RoI 3D AP；报告在线掩码+生成+检测总耗时。原 RHGM 的离线预生成开销要单列，不能只拿其 detector 耗时与在线方法相比。扩散若无可重复的检测增益，先缩小目标区域和点数；仍无收益则停止增加模块。

提交分开做：`model: add online xyz generator`、`model: preserve detector gradient to generated point features`、`eval: add online latency and inference input audit`。每次提交前运行一次小批次前向、反向与 val 子集推理；全量评测通过后再记录结论。

## 3. 置信度融合与联合优化（10 月 26 日–11 月 12 日）

仅在在线生成优于简单补点时进行。生成器输出每点几何置信度 `q∈[0,1]`，用教师几何误差校准。原始雷达与生成点分别编码或分别汇聚到 BEV，门控只作用于生成分支；按位置与置信度控制其贡献，原始雷达分支保留。避免普通无条件 attention。联合损失为 `L_det + λ L_diff + μ L_geom + ν L_conf`；记录各项尺度和梯度，避免检测项淹没几何项。

四格：原 HGSFusion；仅扩散；原 RHGM + 新融合；扩散 + 新融合。第三格的置信度由原 RHGM 可取得的掩码分数和雷达邻距估计，定义及训练预算要与第四格明确。另测 2/4/8 步、点数和误检。用至少 3 个固定种子报告均值及波动；只有增益超过运行波动且额外耗时可接受才保留复杂设计。

提交：`model: add geometry confidence propagation in BEV`；实验结果按统一表格写入 `experiments/online_geometry/README.md`。所有 checkpoint 和逐帧预测写到已忽略的 `output/`，结果表引用 checkpoint 哈希。

## 4. TJ4D 迁移与第二篇独立问题（10 月下旬–11 月底）

TJ4D 没有已验证的独立 LiDAR，不能在其上执行 LiDAR 教师训练。在线生成模型可在 VoD 训练后以雷达/相机输入迁移至 TJ4D，或仅测无教师泛化；结果按 TJ4D AP40、dark/standard/shiny 子集单独报告。不得与 VoD 指标平均。

第二篇从固定基线开 `research/camera-reliability`，只研究相机亮度、模糊、遮挡及夜间条件下的模态可靠性。方法、训练数据、消融与第一篇的生成点置信度门控分开；其四格应围绕退化类型/程度、相机质量估计及模态权重，不复用第一篇的生成点作为贡献。若在同一仓库开发，两个分支各自写独立 `experiments/<topic>/README.md`；共同的基线修复只从 `research/baseline-freeze` 合入，不互相 cherry-pick 论文核心模块。

## 5. 论文与 Git 发布节奏

| 时间 | 交付物 | 决策 |
| --- | --- | --- |
| 10 月 9 日前 | 基线代码版本、权重哈希、数据清单、复测 | 固定可比较对照 |
| 10 月 15 日前 | VoD 教师叠图与在线前景输入 | 对齐合格才训练生成 |
| 10 月 25 日前 | 在线扩散对简单补点的收益与耗时 | 无收益则缩小设计 |
| 11 月上旬 | 第二篇相机退化实验和初稿 | 有独立结论才投稿 |
| 11 月中旬 | 第一篇四格消融、置信度校准与初稿 | 选择最简有效版本 |
| 12 月 1 日前 | 两篇互不重叠的终稿与投稿材料 | 投稿前核验会议资格 |

当前官网列出的 IPAS 2027 截稿为 **2026-11-10**、MOST 2027 为 **2026-12-01**；IV 2027 页面曾列 **2026-11-15**，但另一官方公告提过 10 月 1 日，须临投稿再次查正式 CFP 和系统。提交前分别核对往届 Ei Compendex 记录、正式论文集和学校对录用通知的规则；录用时间不能由截稿时间推断。

代码审阅稳定后可推送个人远端研究分支：`git push -u origin research/online-geometry` 和 `git push -u origin research/camera-reliability`。推送前检查 `git status --short`、`git diff --check`、`git ls-files | grep -E '(\.pth|\.pkl|\.bin|\.tar\.gz)$'`，确保无权重或原始数据。论文复现实验对应的 commit 用带注释 tag 固定；不要因新实验重写旧 tag。
