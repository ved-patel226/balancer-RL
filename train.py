import gym
from stable_baselines3 import PPO
import cv2
import numpy as np
from common import *

env = gym.make("CartPole-v1")


initial_lr = 0.001
final_lr = 0.00001
total_timesteps = 100000

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

callback = TrainAndLoggingCallback(check_freq=20000, verbose=2, save_path=CHECKPOINT_DIR)
model = PPO('MlpPolicy', env, verbose=2, tensorboard_log=LOG_DIR, learning_rate=linear_schedule(0.001), n_steps=256)

#tensorboard --logdir=logs
model.learn(total_timesteps=total_timesteps, callback=callback)

model.save('super balancer')