from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import gym
import math

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, log_interval=10000, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.log_interval = log_interval
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        if self.num_timesteps % self.log_interval == 0:
            self.logger.record('timesteps', self.num_timesteps)
        return True

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

def cosine_schedule(initial_value, final_value, total_steps):
    def func(progress):
        return final_value + (initial_value - final_value) * (1 + math.cos(math.pi * progress)) / 2
    return lambda step: func(step / total_steps)

class SimplifiedControlWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SimplifiedControlWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(4)
    
    def step(self, action):
        action = action[0] if isinstance(action, np.ndarray) else action

        # Default action: no steering, full throttle, no brake
        steering = 0.0
        throttle = 0.0
        brake = 0.0

        if action == 0:  # Full left
            steering = -1.0
        elif action == 1:  # Full right
            steering = 1.0
        elif action == 2:  # Forward, no brake
            throttle = 1.0
        elif action == 3:  # Brake
            brake = 1.0
        else:
            raise ValueError("Invalid action")

        # Map to the continuous action space
        continuous_action = np.array([steering, throttle, brake])
        return self.env.step(continuous_action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)  


if __name__ == '__main__':
    pass