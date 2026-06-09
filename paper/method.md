# Method

The environment is an `8 x 8` grid rendered as RGB frames. The agent begins left of a vertical wall. The visually direct path moves right toward the goal but is blocked by the real transition function. A longer detour through a door is valid.

For each `N`, we sample candidate futures conditioned on action sequences. Some candidates follow the transition function. Others are visually plausible but action-inconsistent: they ghost through the wall, emerge from an occluder without a valid crossing, or move before the corresponding action.

Internal scores combine visual plausibility, temporal smoothness, goal-frame similarity, and a learned-video score. Real utility is never read from the generated video; it is measured by executing the selected action sequence in the ground-truth dynamics.

Repairs are evaluated as a ladder: action-consistency filter, temporal-causality filter, occlusion uncertainty screen, frame-state consistency check, and small held-out video-real calibration.
