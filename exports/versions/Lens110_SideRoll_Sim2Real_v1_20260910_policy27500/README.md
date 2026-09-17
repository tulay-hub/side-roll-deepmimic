# Lens110 SideRoll Sim2Real v1 - policy_27500

本包是 159 维 URDF 顺序策略的侧滚（side_roll）真机部署包，结构与 Dance v6 部署包一致。

策略来源：

```text
lens110RL/lens110/legged_lab_lbot/logs/rsl_rl/lens110_deepmimic/2026-09-10_11-24-24_side_roll_100hz_4096_v6_robust/model_27500.pt
```

训练任务：

```text
LeggedLab-Isaac--Deepmimic-Lens110-SideRoll-v0
num_envs: 4096
observation: 159D
action: 21D
数据: Seed G1 side_roll_R_002__A415_M 重定向 (120Hz -> 100Hz)
鲁棒域随机: 摩擦 0.5~1.5, 骨盆质量 ±0.5kg, PD ±10%, COM x/y ±5cm
```

## ONNX 部署文件

```text
policy_27500/policy.onnx
```

`policy.pt` 是原始 PyTorch checkpoint，只用于追溯和再导出。

## 部署合同

```text
input:  [1, 159]
output: [1, 21]
obs normalization: 已烘焙进 ONNX
action mode:       reference residual
action scale:      0.25
action clip:       null
physics:           500 Hz
policy:            100 Hz
PD:                髋/膝 40/5, 踝 55/5, 腰 100/5, 肩/肘 20/5
```

## 动作数据

```text
motions/side_roll_R_002__A415_M_100hz.npz
```

关节顺序为 USD 交叉顺序，root 四元数为 wxyz，与 Dance v6 的 NPZ 格式一致。

## MuJoCo 回放验证

```bash
cd replay
python play_lens110_official_159_urdf.py \
    --onnx ../policy_27500/policy.onnx \
    --motion ../motions/side_roll_R_002__A415_M_100hz.npz \
    --mjcf mjcf/lens110_21dof.xml \
    --deploy ../config/deploy_config_159_urdf.yaml
```

## 包完整性检查

```bash
python scripts/check_package.py
```
