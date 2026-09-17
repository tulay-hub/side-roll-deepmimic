# 翻滚 SideRoll 奖励结构

SideRoll 继承 159-D DeepMimic 参考跟踪奖励，专门修改终止和滑动约束：滚地允许横躺和躯干接触，关闭
`base_contact`/`bad_orientation`，`base_height` 下限为 `0.02`；根/关键点偏差阈值为 `0.8 m`，
`alive=0.05`，并在参考静止段使用 `standing_still=-2.0` 防止蹭步。

代码在共享基座的
`framework/isaaclab_shared/lens110/legged_lab_lbot/source/legged_lab/legged_lab/tasks/locomotion/deepmimic/config/lens110/lens110_deepmimic_env_cfg_sideroll.py`。
完整结构见 [`docs/REWARD_FRAMEWORKS.md`](../../../docs/REWARD_FRAMEWORKS.md)。
