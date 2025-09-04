import numpy as np
import os
import gymnasium as gym
import time
import argparse
import mujoco
import yaml
from stable_baselines3 import PPO

import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
import mimoEnv.utils as env_utils
import babybench.utils as bb_utils
import babybench.eval as bb_eval

from learn_selftouch.intrinsic_selftouch_count import Wrapper

import matplotlib.pyplot as plt

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='learn_selftouch/config_selftouch.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--render', default=True,  type=bool,
                        help='Renders a video for each episode during the evaluation.')
    parser.add_argument('--duration', default=1000, type=int,
                        help='Total timesteps per evaluation episode')
    parser.add_argument('--episodes', default=10, type=int,
                        help='Number of evaluation episode')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = bb_utils.make_env(config, training=False)
    env = Wrapper(env)
    env.reset()

    # Initialize evaluation object
    evaluation = bb_eval.EVALS[config['behavior']](
        env=env,
        duration=args.duration,
        render=args.render,
        save_dir=config['save_dir'],
    )

    # Preview evaluation of training log
    evaluation.eval_logs()
    model_dir=config['save_dir']
    print(f"Model dir: {model_dir}")
    model=PPO.load(model_dir+'/model')

    for ep_idx in range(args.episodes):
        print(f'Running evaluation episode {ep_idx+1}/{args.episodes}')

        # Reset environment and evaluation
        obs, _ = env.reset()
        evaluation.reset()

        # Store an array of habituation values and sample it
        # every 20 timesteps during the episode to make a plot
        # of the habituation over the episode.
        habituation = []

        for t_idx in range(args.duration):

            # Select action
            #action = env.action_space.sample()
            # Append to habituation array if step is a multiple
            # of 20.
            if t_idx % 20 == 0:
                habituation.append(env.habituation)

            action, next_state = model.predict(obs)

            # ---------------------------------------------------# 
            #                                                    #
            # TODO REPLACE WITH CALL TO YOUR TRAINED POLICY HERE #
            # action = policy(obs)                               #
            #                                                    #
            # ---------------------------------------------------#

            # Perform step in simulation
            obs, _, _, _, info = env.step(action)

            # Perform evaluations of step
            evaluation.eval_step(info)
            
        evaluation.end(episode=ep_idx)

        # Make a plot of the habituation. We have 22 body parts, so
        # we make a 5x5 plot.
        habituation_names = env.body_names
        habituation = np.array(habituation)
        x_vals = np.array(range(len(habituation)))
        x_vals *= 20  # Times step size

        fig, axs = plt.subplots(5, 5, figsize=(10, 8))

        for i in range(len(habituation_names)):
            x_pos = i % 5
            y_pos = i // 5
            axs[x_pos, y_pos].plot(x_vals, habituation[:,i],
                                   color='green')
            axs[x_pos, y_pos].set_title(habituation_names[i])

        plt.tight_layout()
        plt.savefig('habplot.png')


if __name__ == '__main__':
    main()
