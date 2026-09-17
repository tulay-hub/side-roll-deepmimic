# 05 · 翻滚（Side Roll）

## 项目定位

侧滚是 DeepMimic 的专用 159-D URDF-order 任务，不是普通全身舞蹈动作改名。

- 训练任务：`LeggedLab-Isaac--Deepmimic-Lens110-SideRoll-v0`
- PLAY 任务：`LeggedLab-Isaac--Deepmimic-Lens110-SideRoll-Play-v0`
- policy：`159 -> 21`
- 参考动作：`side_roll_R_002__A415_M_100hz_lens110`
- 导出包：`exports/versions/Lens110_SideRoll_Sim2Real_v1_20260910_policy27500`

## 目录

```text
05_side_roll/
├── framework/isaaclab_shared -> framework/isaaclab_shared
├── data/motions/side_roll_retargeted/  # 动作5：csv/npy/pkl/txt/100 Hz 结果
├── data/training/deepmimic_motion -> shared motion directory
├── exports/versions/                   # 侧滚 sim2real 包
├── exports/training_exports -> SideRoll runs
├── experiments/deepmimic_runs -> shared DeepMimic logs
└── docs/
```

## 训练

```bash
./scripts/train.sh \
  --headless --num_envs 4096 --max_iterations 30000
```

## 专用奖励/终止

滚地阶段允许横躺和躯干接触，因此 `base_contact`、`bad_orientation` 被关闭，`base_height` 下限降到 `0.02`；
根/关键点偏差收紧到 `0.8 m`，并加入参考静止段的 `standing_still=-2.0`，避免站立阶段蹭步。摩擦随机
`0.5..1.5`、骨盆质量 `+-0.5 kg` 用于覆盖滑动条件。完整说明见
[`docs/REWARD_FRAMEWORKS.md`](docs/REWARD_FRAMEWORKS.md)。

## English

This repository contains the side-roll motion project. It uses DeepMimic reference-motion imitation with the verified 159-dimensional URDF-order policy contract. Roll-specific contact, posture, friction, and termination behavior is intentionally separate from ordinary walking.

Run `./scripts/train.sh --headless --num_envs 4096` in the local Isaac Lab environment. Keep side-roll motions, replay MJCF, policy exports, mesh assets, and validation reports under the project directories. Before deployment, verify the 159-dimensional order, quaternion convention, contact geometry, action scale, and simulation-versus-hardware boundary.

The common reward and interface contracts are in `docs/REWARD_FRAMEWORKS.md` and `docs/INTERFACE_CONTRACTS.md`.
