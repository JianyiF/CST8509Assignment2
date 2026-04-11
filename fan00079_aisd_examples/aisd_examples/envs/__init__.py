from gymnasium.envs.registration import register, registry

env_id = "aisd_examples/CreateRedBall-v0"

if env_id not in registry:
    register(
        id=env_id,
        entry_point="aisd_examples.envs.create3_red_ball:CreateRedBallEnv",
    )