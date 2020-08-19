import numpy as np

from tf_agents.environments import py_environment as pyenv, \
								 tf_py_environment, utils
from tf_agents.specs import array_spec 
from tf_agents.trajectories import time_step

from MinerEnv import MinerEnv

MAP_MAX_X = 21 #Width of the Map
MAP_MAX_Y = 9  #Height of the Map
K = 3

class TFAgentsMiner(pyenv.PyEnvironment):
	def __init__(self, host, port, debug = False):
		super(TFAgentsMiner, self).__init__()

		self.miner_env= MinerEnv(host, port)
		self.miner_env.start()
		self.debug = debug
		
		self._action_spec = array_spec.BoundedArraySpec(shape = (), dtype = np.int32, minimum = 0, maximum = 5, name = 'action')
		self._observation_spec = array_spec.BoundedArraySpec(shape = (MAP_MAX_X*5,MAP_MAX_Y*5,6), 
			dtype = np.float32, name = 'observation')

	def action_spec(self):
		return self._action_spec

	def observation_spec(self):
		return self._observation_spec

	def _reset(self):
		mapID = np.random.randint(1, 6)
		posID_x = np.random.randint(MAP_MAX_X)
		posID_y = np.random.randint(MAP_MAX_Y)
		request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100")
		self.miner_env.send_map_info(request)
		self.miner_env.reset()
		observation = self.miner_env.get_state()

		return time_step.restart(observation)

	def _log_info(self):
		info = self.miner_env.socket

		# print(f'Map size:{self.info.user.max_x, self.miner_env.state.mapInfo.max_y}')
		print(f"Self  - Pos ({info.user.posx}, {info.user.posy}) - Energy {info.user.energy} - Status {info.user.status}")
		for bot in info.bots:
			print(f"Enemy  - Pos ({bot.info.posx}, {bot.info.posy}) - Energy {bot.info.energy} - Status {bot.info.status}")
				
	def _step(self, action):
		if self.debug:
			self._log_info()
			
		self.miner_env.step(str(action))
		observation = self.miner_env.get_state()
		reward = self.miner_env.get_reward()

		if not self.miner_env.check_terminate():
			return time_step.transition(observation, reward)
		else:
			self.reset()
			return time_step.termination(observation, reward)

	def render(self):
		pass

if __name__ == '__main__':
	env = TFAgentsMiner("localhost", 1111)
	utils.validate_py_environment(env, episodes=5)