from my_env import EvidenceEnv
import agents
import matplotlib.pyplot as plt


def runAgent():
	# reset environment and agent
	obs = env.reset()
	reward = None
	done = False
	R = []
	cum_R = []
	for step in range(n_iter):
		env.render()
		action = agent.act(obs)
		# print("Action: " + str(action))
		_obs, reward, done, _ = env.step(action)

		# no training involved for random agent
		agent.train(action, obs, reward, _obs)
		obs = _obs
		R.append(reward)

	for r in range(len(R)):
		cum_R.append(sum(R[:r]))

	plt.plot(cum_R)
	plt.xlabel("time")
	plt.ylabel("Cumulative Reward")
	plt.show()


def plotQ(Q):
	states = [[0, 0], [0, 1], [1, 0], [1, 1]]
	for state in states:
		for a in [0, 1]:
			print("Q[{},{}]={}".format(state, a, Q[env.asint(state), a]))


# Number of iterations
n_iter = 1000

# environment specs
env = EvidenceEnv(n=2, p=0.75)

agent = agents.RandomAgent(env)
runAgent()

# define agent
agent = agents.TabularQAgent(env)
plotQ(agent.Q)
runAgent()
plotQ(agent.Q)

actualQ = agent.Q
agent = agents.NeuralAgent(env, actualQ)
plotQ(agent.Q)
runAgent()
plotQ(agent.Q)