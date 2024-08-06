import gym
from stable_baselines3 import PPO
from termcolor import cprint

env = gym.make("CartPole-v1")
state = env.reset()

model = PPO.load(r"super balancer.zip")

while True:
    state_copy = state.copy()
    action, _state = model.predict(state_copy)
    state, reward, done, info = env.step(action)
    env.render()