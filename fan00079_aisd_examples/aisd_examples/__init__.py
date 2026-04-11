from gymnasium.envs.registration import register

register(
    id="aisd_examples/CreateRedBall-v0",
    entry_point="aisd_examples.envs.create3_red_ball:CreateRedBallEnv",
)