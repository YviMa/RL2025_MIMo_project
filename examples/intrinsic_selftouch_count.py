
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
        
        self.body_names=self.env.touch_params['scales'].keys()
        print("body_names",len(self.body_names))
        # redefine obs space
        old_dict=self.env.observation_space.spaces
        new_dict=old_dict.copy()
        #adding a new box with adjusted size
        #size is equal to the number of body parts 
        new_dict['touch']=gym.spaces.Box(-np.inf, np.inf, shape=(len(self.body_names),), dtype=np.float32)
        new_dict['habituation']=gym.spaces.Box(-np.inf, np.inf, shape=(len(self.body_names),), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(new_dict)
        self.h_tracker=np.zeros(len(self.body_names))
        self.reward_tracker=np.ones(len(self.body_names))
        self.tau_h=1
        self.tau_d=1


    def compute_intrinsic_reward(self, obs):
        #intrinsic_reward = np.sum(obs['touch'] > 1e-6) / len(obs['touch'])
        intrinsic_reward=np.ones(len(obs['touch']))
        intrinsic_reward[self.h_tracker!=0]=self.reward_tracker[self.h_tracker!=0]-1/self.tau_h*np.exp(-1/self.tau_h)
        intrinsic_reward[self.d_tracker==0]=self.reward_tracker[self.h_tracker==0]+1/self.tau_d*np.exp(-1/self.tau_d)
        self.reward_tracker=intrinsic_reward
        return intrinsic_reward

    def step(self, action):
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        # redefine obs
        touch_setup=self.env.touch
        sensor_outputs=touch_setup.sensor_outputs

        body_dict={}
        for body_name in self.body_names:
            body_dict.update({body_name:self.env.model.body(body_name).id})


        obs_touch = np.zeros(len(self.body_names))
        for idx, body_part in enumerate(self.body_names):
            obs_touch[idx]=np.any(sensor_outputs[body_dict[body_part]]) 

        obs['touch']=obs_touch
        
        prev_habituation=obs['habituation']
        new_habituation=np.zeros(np.shape(prev_habituation))
        new_habituation[obs_touch==1]=prev_habituation[obs_touch==1]-self.hab(prev_habituation[obs_touch==1]) #habituation where there is touch
        new_habituation[obs_touch==0]=prev_habituation[obs_touch==0]-self.dehab(prev_habituation[obs_touch==0])  #dehabituation where there is no touch
        obs['habituation']=new_habituation
             
        #compute reward from redefined observation
        intrinsic_reward = self.compute_intrinsic_reward(obs)
        total_reward = intrinsic_reward + extrinsic_reward # extrinsic reward is always 0  

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info=self.env.reset(**kwargs)
        obs['touch']=np.zeros(len(self.body_names),dtype=np.float32)
        obs['habituation']=np.ones(len(self.body_names),dtype=np.float32)
        self.h_tracker=np.zeros(len(self.body_names))
        self.d_tracker=np.zeros(len(self.body_names))

        return obs, info
    
    def hab(self,y):
        # performs habituation step
        # habituation e^{-x/tau} => grad -1/tau => ln(y)=-x/tau => -tau*ln(y)=x
        # y= -1/tau*e^(-xt/tau) => ln(-tau*y)
        x=-self.tau_h*np.log(y)
        grad=-1/self.tau_h*e^(-x/self.tau_h)
        new_hab=y+grad
        return new_hab
    
    def dehab(self,y):
        # performs dehabituation step
        x=-self.tau_d*np.log(-y) #1-e^{-x/\tau_d}=y => ln(-y)*(-tau)=x 
        grad=1/self.tau_d*e^(-x/self.tau_h) #1/tau
        new_hab=y+grad
        return new_hab

    


def main():

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

    model = PPO("MultiInputPolicy", wrapped_env, verbose=1,learning_rate=0.005)
    model.learn(total_timesteps=args.train_for)

    model.save(os.path.join(config["save_dir"], "model"))

    env.close()

def analysis():
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
    test=wrapped_env.override_observation_space()  
    print(test)

if __name__ == '__main__':
    #analysis()
    main()

