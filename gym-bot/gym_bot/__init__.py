from gym.envs.registration import register
register(
	id="advbot-v6",
	entry_point='gym_bot.envs:AdvBotEnvSingleDetectLargeHiar'
)