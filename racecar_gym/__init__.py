from gym.envs.registration import register

register(
     id="RaceCar-v0",
     entry_point="racecar_gym.envs:RaceCarEnv",
)
