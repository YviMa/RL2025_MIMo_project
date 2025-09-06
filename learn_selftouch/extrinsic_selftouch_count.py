
"""
Example: Maximize number of touches for self-touch.
"""
import numpy as np
import os
import gymnasium as gym
import time
import argparse
import mujoco
import yaml
from stable_baselines3 import PPO
import sys
sys.path.append(".")
sys.path.append("..")
import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
import mimoEnv.utils as env_utils
import babybench.utils as bb_utils
import matplotlib.pyplot as plt

class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # Array of body part names. see mimoEnv/babybench/selftouch.py
        self.body_names = np.concatenate([np.array(env_utils.get_geoms_for_body(self.model, body_id)) for body_id in self.mimo_bodies])
        # redefine obs space
        old_dict=self.env.observation_space.spaces
        new_dict=old_dict.copy()
        #adding a new box with adjusted size
        #size is equal to the number of body parts 
        new_dict.update({'reward':gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)})   
        new_dict.update({'touched_parts_left_hand':gym.spaces.MultiBinary(len(self.body_names))})
        new_dict.update({'touched_parts_right_hand':gym.spaces.MultiBinary(len(self.body_names))})
        self.observation_space = gym.spaces.Dict(new_dict)

        self.extr_left_touches = set()
        self.extr_right_touches = set()
    
    def compute_extrinsic_reward(self, info):
        #touches = len(info['right_hand_touches']) + len(info['left_hand_touches'])
        #return touches / (len(self.body_names) * 2)

        #get array of previously or now touched body parts
        right = set(info['right_hand_touches'])
        left = set(info['left_hand_touches'])

        #set operations to find newly touched body parts
        changed = len(right - self.extr_right_touches)
        changed += len(left - self.extr_left_touches)

        #update sets of previously touched body parts
        self.extr_right_touches.update(right)
        self.extr_left_touches.update(left)

        #print newly touched body parts
        #reward=newly touched body parts
        if changed > 0:
            print(f"New body parts touched! Left: {self.extr_left_touches}, Right: {self.extr_right_touches}")
        return changed
    
    def get_binary_touch_arrays(self):
        left_touch_array = np.zeros(len(self.body_names), dtype=int)
        right_touch_array = np.zeros(len(self.body_names), dtype=int)

        #map touched body parts to binary array
        # -3 because body part ids [3,38]
        for left_touch in self.extr_left_touches:
            left_touch_array[left_touch - 3] = 1
        for right_touch in self.extr_right_touches:
            right_touch_array[right_touch - 3] = 1

        return left_touch_array, right_touch_array

    def step(self, action):
        #should implement hot coding of touched body parts as part of the observation
        #random action for baseline: self.env.action_space.sample()
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        extrinsic_reward = self.compute_extrinsic_reward(info)
        total_reward = extrinsic_reward 

        #add reward to state
        obs.update({'reward':np.array([total_reward],dtype=np.float32)})

        #add touched body parts to state as binary arrays
        left_array, right_array = self.get_binary_touch_arrays()
        obs.update({'touched_parts_left_hand': left_array})
        obs.update({'touched_parts_right_hand': right_array})

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info=self.env.reset(**kwargs)
        obs['reward']=np.zeros(1,dtype=np.float32)
        obs['touched_parts_left_hand']=np.zeros(len(self.body_names), dtype=int)
        obs['touched_parts_right_hand']=np.zeros(len(self.body_names), dtype=int)

        #continue with touches from last episode:
        #self.extr_left_touches = set()
        #self.extr_right_touches = set()
        return obs, info


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='learn_selftouch/config_selftouch.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--train_for', default=10000, type=int,
                        help='Total timesteps of training')
    args = parser.parse_args()
    
    with open(args.config) as f:
            config = yaml.safe_load(f)

    env = bb_utils.make_env(config)
    wrapped_env = Wrapper(env)
    wrapped_env.reset()

    model = PPO("MultiInputPolicy", wrapped_env, verbose=1,learning_rate=0.005)
    model.learn(total_timesteps=args.train_for)

    model.save(os.path.join(config["save_dir"], "model"))

    env.close()

def test_hab():
    # testing habituation
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='examples/config_selftouch.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--train_for', default=10000, type=int,
                        help='Total timesteps of training')
    args = parser.parse_args()

    with open(args.config) as f:
            config = yaml.safe_load(f)

    env = bb_utils.make_env(config)
    wrapped_env = Wrapper(env)
    wrapped_env.reset()

    x=np.arange(0,10)
    target=np.exp(-x/wrapped_env.tau_h)
    wrapped_env.habituation=np.ones(len(wrapped_env.body_names))

    for n in range(0,10):
        new_hab=wrapped_env.hab(wrapped_env.habituation)
        assert np.all(np.abs(target[n]-wrapped_env.habituation)<10**(-6))
        wrapped_env.habituation=new_hab

def test_dehab():
    # testing dehabituation
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='examples/config_selftouch.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--train_for', default=10000, type=int,
                        help='Total timesteps of training')
    args = parser.parse_args()

    with open(args.config) as f:
            config = yaml.safe_load(f)

    env = bb_utils.make_env(config)
    wrapped_env = Wrapper(env)
    wrapped_env.reset()

    x=np.arange(10)
    target=1-np.exp(-x/wrapped_env.tau_d)
    wrapped_env.habituation=np.zeros(len(wrapped_env.body_names))

    for n in range(10):
        new_hab=wrapped_env.dehab(wrapped_env.habituation)
        assert np.all(np.abs(target[n]-wrapped_env.habituation)<10**(-6))
        wrapped_env.habituation=new_hab

def test_compute_intrinsic_reward():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='examples/config_selftouch.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--train_for', default=10000, type=int,
                        help='Total timesteps of training')
    args = parser.parse_args()

    with open(args.config) as f:
            config = yaml.safe_load(f)

    env = bb_utils.make_env(config)
    wrapped_env = Wrapper(env)
    wrapped_env.reset()
    wrapped_env.habituation=np.random.uniform(0,1,len(wrapped_env.body_names))
    obs_touch=np.zeros(len(wrapped_env.body_names))
    obs_touch[0:6]=1
    obs={'touch':obs_touch}
    assert (wrapped_env.compute_intrinsic_reward(obs)-np.sum(wrapped_env.habituation[obs_touch.astype(bool)]))<10**(-6)

if __name__ == '__main__':
    main()

