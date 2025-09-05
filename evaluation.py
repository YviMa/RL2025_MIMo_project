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

        # Store habituation values per body-part. For this, simply use the average habituation
        # of all habituation values of sensors for that body part.
        habituations = []

        for t_idx in range(args.duration):

            # Store habituation every 20 steps.
            if t_idx % 20 == 0:
                habituation = env.habituations.copy()
                for key in habituation.keys():
                    habituation[key] = np.mean(habituation[key])

                habituations.append(habituation)

            # Select action
            #action = env.action_space.sample()

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

        # Save plot of habituation values per body-part.
        fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(12, 12))
        body_names = env.touch_params['scales']
        timesteps = np.array(range(len(habituations))) * 20

        for i in range(5):
            for j in range(5):
                idx=5*i+j
                if idx >= len(body_names): return

                body_name = body_names[idx]
                body_id = env.model.body(body_name).id

                ax = axes[i, j]
                habs = []
                for i in habituations:
                    habs.append(i[body_id])
                ax.plot(timesteps, habs, color='green')
                ax.set_title(body_name)

        plt.tight_layout()
        plt.savefig('habplot.png')


if __name__ == '__main__':
    main()
