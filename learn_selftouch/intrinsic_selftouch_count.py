
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
        
        # Array of body part names.
        self.body_names=self.env.touch_params['scales'].keys()
        # redefine obs space
        old_dict=self.env.observation_space.spaces
        new_dict=old_dict.copy()
        #adding a new box with adjusted size
        #size is equal to the number of body parts 
        new_dict['touch']=gym.spaces.Box(-np.inf, np.inf, shape=(len(self.body_names),), dtype=np.float32)
        new_dict.update({'habituation':gym.spaces.Box(-np.inf, np.inf, shape=(len(self.body_names),), dtype=np.float32)})
        new_dict.update({'reward':gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)})   
        self.observation_space = gym.spaces.Dict(new_dict)

        self.habituation = np.ones(len(self.body_names))
        # habituation time constants. reward after touch decays with function -exp(t/tau_h) and
        # recovers with function 1-exp(t/tau_d).
        self.tau_h=1
        self.tau_d=1


    def compute_intrinsic_reward(self, obs):
        # Use metabolic cost as a penalty, clipped at 0.1
        #metabolic_cost = self.env.actuation_model.cost()
        #metabolic_cost = np.clip(metabolic_cost, 0, 0.1)

        # Use 'obs['touch']' as mask to filter out body parts that are being touched. 
        mask = obs['touch'].astype(bool)

         # The reward is then the sum of habituations over touched body parts minus penalty
        return np.sum(self.habituation[mask])  #- metabolic_cost

    def step(self, action):
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        # redefine obs
        touch_setup=self.env.touch
        # dictionary that has keys id of the body part and values array of the sensor outputs.
        sensor_outputs=touch_setup.sensor_outputs

        # dictionary that assigns each body part name the id of that body part.
        body_dict={}
        for body_name in self.body_names:
            body_dict.update({body_name:self.env.model.body(body_name).id})

        # Array of sensor observations. Value is True if that body part is touched and else False.
        # We check if a body part is touched by checking if any sensor of that body part is active
        # by a threshold.
        obs_touch = np.zeros(len(self.body_names))
        for idx, body_part in enumerate(self.body_names):
            obs_touch[idx]=np.any(sensor_outputs[body_dict[body_part]]>10**(-6)) 

        obs['touch']=obs_touch
        
        prev_habituation=self.habituation
 
        new_habituation=np.zeros(np.shape(prev_habituation))
        new_habituation[obs_touch==1]=self.hab(prev_habituation[obs_touch==1]) #habituation where there is touch
        #new_habituation[obs_touch==0]=self.dehab(prev_habituation[obs_touch==0])  #dehabituation where there is no touch
        new_habituation[obs_touch==0]=prev_habituation[obs_touch==0]
        obs.update({'habituation':new_habituation})
             
        #compute reward from redefined observation
        intrinsic_reward = self.compute_intrinsic_reward(obs)
        total_reward = intrinsic_reward + extrinsic_reward # extrinsic reward is always 0  
        self.habituation=new_habituation

        #add reward to state
        obs.update({'reward':np.array([total_reward],dtype=np.float32)})

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info=self.env.reset(**kwargs)
        obs['touch']=np.zeros(len(self.body_names),dtype=np.float32)
        obs['habituation']=np.ones(len(self.body_names),dtype=np.float32)
        obs['reward']=np.zeros(1,dtype=np.float32)
        return obs, info
    
    def hab(self,y):
        # performs habituation step
        # habituation e^{-x/tau} => grad -1/tau => ln(y)=-x/tau => -tau*ln(y)=x
        # y= -1/tau*e^(-xt/tau) => ln(-tau*y)
        x=np.ones(np.shape(y))
        x[y>10**(-6)]=-self.tau_h*np.log(y[y>10**(-6)])
        new_hab=np.zeros(np.shape(y))
        # TODO adjust x+1 to the time step that we want to take?
        new_hab[y>10**(-6)]=np.exp(-(x[y>10**(-6)]+1)/self.tau_h)
        new_hab[y<=10**(-6)]=0
        return new_hab
    
    def dehab(self,y):
        # performs dehabituation step
        x=np.zeros(np.shape(y))
        x[y!=1]=-self.tau_d*np.log(1-y[y!=1]) #1-e^{-x/\tau_d}=y => ln(1-y)*(-tau)=x
        new_hab=np.zeros(np.shape(y)) 
        new_hab[y!=1]=1-np.exp(-(x[y!=1]+1)/self.tau_d)
        new_hab[y==1]=1
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

