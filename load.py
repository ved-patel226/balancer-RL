import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from termcolor import cprint

from gym.wrappers import FrameStack, GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import matplotlib.pyplot as plt

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

model = PPO.load(r"failed_train\best_model_390000.zip")

state = env.reset()

while True:
    action, _state = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()