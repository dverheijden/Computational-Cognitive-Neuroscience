from my_env import EvidenceEnv
import agents
import matplotlib.pyplot as plt

# Number of iterations
n_iter = 1000

# environment specs
env = EvidenceEnv(n=2, p=0.75)

# define agent
# agent = agents.TabularQAgent(env)
agent = agents.NeuralAgent(env)
# reset environment and agent
obs = env.reset()
reward = None
done = False
R = []
cum_R = []
for step in range(n_iter):
	env.render()
	action = agent.act(obs)
	print("Action: " + str(action))
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
