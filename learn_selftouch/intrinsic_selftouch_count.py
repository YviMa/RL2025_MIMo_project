
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
    def __init__(self, env, componentwise=False, habituation_reset=False, reward_state=False):
        super().__init__(env)
        
        # Array of body part names.
        self.body_names=np.concatenate([np.array(env_utils.get_geoms_for_body(self.model, body_id)) for body_id in self.mimo_bodies])
        # redefine obs space
        old_dict=self.env.observation_space.spaces
        new_dict=old_dict.copy()

        self.n_sensors=int(old_dict['touch'].shape[0]/3)
        self.componentwise=componentwise
        self.habituation_reset=habituation_reset
        self.reward_state=reward_state
        self.init_habituation_dict()

        if self.reward_state:
            new_dict.update({'reward':gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)})   
        
        if self.componentwise:
            new_dict.update({'habituation':gym.spaces.Box(-np.inf, np.inf, shape=old_dict['touch'].shape, dtype=np.float32)})
            self.observation_space = gym.spaces.Dict(new_dict)
        else:
            new_dict.update({'touch':gym.spaces.Box(-np.inf, np.inf, shape=(self.n_sensors,), dtype=np.float32)})
            new_dict.update({'habituation':gym.spaces.Box(-np.inf, np.inf, shape=(self.n_sensors,), dtype=np.float32)})
            self.observation_space = gym.spaces.Dict(new_dict)
        # habituation time constants. reward after touch decays with function -exp(t/tau_h) and
        # recovers with function 1-exp(t/tau_d).
        self.tau_h=1
        self.tau_d=1

    def init_habituation_dict(self):
        """ Returns inited habituation dictionary. """
        self.habituation = {}

        if self.componentwise:
            for geom_id in self.env.touch.sensor_positions:
                self.habituation[geom_id] = np.ones((self.env.touch.get_sensor_count(geom_id), 3), dtype=np.float32)

        else:
            for geom_id in self.env.touch.sensor_positions:
                self.habituation[geom_id] = np.ones((self.env.touch.get_sensor_count(geom_id),), dtype=np.float32)

    def compute_intrinsic_reward(self, obs):
        # Use metabolic cost as a penalty, clipped at 0.1
        #metabolic_cost = self.env.actuation_model.cost()
        #metabolic_cost = np.clip(metabolic_cost, 0, 0.1)

        # Use 'obs['touch']' as mask to filter out body parts that are being touched. 
        mask = obs['touch'].astype(bool)

         # The reward is then the sum of habituations over touched body parts minus penalty
        return np.sum(obs['habituation'][mask]) # - metabolic_cost

    def calculate_habituation(self):
        """ Calculates the habituation based on the sensor_output dictionary supplied by
        self.env.touch.sensor_output. """
        thresh = 10**(-6)
        for body_id in self.env.touch.meshes:
            # 'sensor_outputs' is a list of force vectors. Depending
            # on whether 'self.componentwise' is active or not, perform
            # touch check on each component or on the entire vector.
            sensor_outputs = self.env.touch.sensor_outputs[body_id]
            sensor_outputs = sensor_outputs > thresh

            # If not componentwise, check if any of the component exceeded threshold.
            if not self.componentwise:
                sensor_outputs = np.any(sensor_outputs, axis=1)

            self.habituation[body_id][sensor_outputs]=self.hab(self.habituation[body_id][sensor_outputs]) 

    def step(self, action):
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        self.calculate_habituation()
        obs['habituation']=self.env.touch.flatten_sensor_dict(self.habituation)

        if self.componentwise:
            obs_touch=(obs['touch']>10**(-6))
        
        else:
            obs_touch=np.reshape((obs['touch']>10**(-6)),(-1,3))
            obs_touch=np.any(obs_touch, axis=1)

        obs['touch']=obs_touch
             
        #compute reward from redefined observation
        intrinsic_reward = self.compute_intrinsic_reward(obs)
        total_reward = intrinsic_reward + extrinsic_reward # extrinsic reward is always 0  

        #add reward to state
        if self.reward_state:
            obs.update({'reward':np.array([total_reward],dtype=np.float32)})

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info=self.env.reset(**kwargs)

        if self.componentwise:
            obs['touch']=np.zeros(obs['touch'].shape,dtype=np.float32)
        else:
            obs['touch']=np.zeros(self.n_sensors,dtype=np.float32)

        if self.habituation_reset:
            self.init_habituation_dict()
            obs['habituation']=np.ones(obs['touch'].shape,dtype=np.float32)
        else:
            obs['habituation']=self.env.touch.flatten_sensor_dict(self.habituation)

        if self.reward_state:
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

    model = PPO("MultiInputPolicy", wrapped_env, verbose=1)
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

