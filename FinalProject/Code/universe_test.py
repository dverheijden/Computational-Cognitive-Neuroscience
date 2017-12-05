import gym
import atari_py  # register the universe environments
import numpy as np
import random

env = gym.make('Pong-v4')
# env.configure(remotes=1)  # automatically creates a local docker container
observation_n = env.reset()

while True:
  action_n = env.action_space.sample()

  observation_n, reward_n, done_n, info = env.step(action_n)

  
  # if reward_n[0] > 0:
  #   print(observation_n)
  #   exituniverse
  if done_n:
    env.reset()
  env.render()

