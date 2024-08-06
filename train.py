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
from datetime import datetime

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

state = env.reset()

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
unique_log_dir = os.path.join(LOG_DIR, f"PPO_{run_id}")

callback = TrainAndLoggingCallback(check_freq=1000, verbose=2, save_path=CHECKPOINT_DIR)
model = PPO('CnnPolicy', env, verbose=2, tensorboard_log=unique_log_dir, learning_rate=0.000001, n_steps=1000)


model.learn(total_timesteps=1000000, callback=callback, log_interval=1)
model.save('OneMillionSteps')