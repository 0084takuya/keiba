import gym
from gym import spaces
import numpy as np
import pandas as pd

class KeibaEnv(gym.Env):
    def __init__(self, X_path, y_path):
        super().__init__()
        self.X = pd.read_pickle(X_path).values.astype(np.float32)
        self.y = pd.read_pickle(y_path).values.astype(np.float32)
        self.n_samples, self.n_features = self.X.shape
        # 行動空間: 0=賭けない, 1=賭ける
        self.action_space = spaces.Discrete(2)
        # 状態空間: 特徴量ベクトル
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32)
        self.current_idx = 0

    def reset(self):
        self.current_idx = 0
        return self.X[self.current_idx]

    def step(self, action):
        # action: 0=賭けない, 1=賭ける
        reward = 0
        done = False
        info = {}
        if action == 1:
            reward = self.y[self.current_idx]  # 3位以内なら1, それ以外0
        self.current_idx += 1
        if self.current_idx >= self.n_samples:
            done = True
            next_state = np.zeros(self.n_features, dtype=np.float32)
        else:
            next_state = self.X[self.current_idx]
        return next_state, reward, done, info

if __name__ == "__main__":
    env = KeibaEnv('keiba_sanshutsu/deep_rl_analysis/X.pkl', 'keiba_sanshutsu/deep_rl_analysis/y.pkl')
    obs = env.reset()
    total_reward = 0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"step: action={action}, reward={reward}, done={done}")
        if done:
            break
    print(f"total_reward: {total_reward}") 