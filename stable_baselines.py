#import gymnasium as gym
from typing import Callable

import wandb

from SFQenv import SFQ
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from wandb.integration.sb3 import WandbCallback
from pprint import pprint
config = {

    # "env_name": "CartPole-v1",
}

sweep_config = {
    'method': 'random',
    'program':'Educational/stable_baselines.py',
    "policy_type": "MlpPolicy",
    "total_timesteps": int(10e6),
}

metric = {
    'name': 'loss',
    'goal': 'minimize'
}

sweep_config['metric'] = metric

parameters_dict = {
    'ent_coef': {
        'values': [0.00001, 0.0001, 0.001, 0.01]
    },
    'policy_layer_size': {
        'values': [64, 128, 256]
    },
    'value_layer_size': {
        'values': [64, 128, 256]
    },
}

parameters_dict.update({
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
    },
    'batch_size': {
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 6000
    }
})

sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project="SFQ")

run = wandb.init()


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


#
#
# class HParamCallback(BaseCallback):
#     """
#     Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
#     """
#
#     def _on_training_start(self) -> None:
#         hparam_dict = {
#             "algorithm": self.model.__class__.__name__,
#             "gamma": self.model.gamma,
#             "GAE_lambda": self.model.gae_lambda,
#             "entropy_coef": self.model.ent_coef,
#             "batch_size": self.model.batch_size,
#             "n_epochs": self.model.n_epochs,
#             "vf_coef": self.model.vf_coef,
#
#         }
#         # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
#         # Tensorboard will find & display metrics from the `SCALARS` tab
#         metric_dict = {
#             "rollout/ep_len_mean": 0,
#             "rollout/ep_rew_mean": 0.0,
#             "train/value_loss": 0.0,
#             "train/entropy_loss": 0.0,
#             "train/loss": 0.0,
#             "train/approx_kl": 0.0,
#             "train/learning_rate": 0.0,
#
#         }
#         self.logger.record(
#             "hparams",
#             HParam(hparam_dict, metric_dict),
#             exclude=("stdout", "log", "json", "csv"),
#         )
#
#     def _on_step(self) -> bool:
#         return True


# env = gym.make("LunarLander-v2")
pprint(sweep_config)
if __name__=="__main__":
    env = SFQ()
    vec_env = make_vec_env(SFQ, n_envs=14)
    run_config = wandb.config
    wandb.agent(sweep_id, count=100)
    policy = ActorCriticPolicy(net_arch=dict(pi=[run_config.policy_layer_size, run_config.policy_layer_size],
                                             vf=[run_config.value_layer_size, run_config.value_layer_size]))
    model = PPO("MlpPolicy", vec_env, verbose=2, batch_size=run_config.batch_size, ent_coef=run_config.ent_coef,
                learning_rate=linear_schedule(run_config.learning_rate), tensorboard_log=f"new_PPO_runs/")

    # model = PPO.load("model_new.zip", env=vec_env, ent_coef=0.0001, batch_size=6000, learning_rate=linear_schedule(0.00006))
    model.learn(
        total_timesteps=config["total_timesteps"],  # tb_log_name=f"new_PPO_runs/{run.id}",
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/",
            verbose=2,
        ),
    )
    # model.learn(total_timesteps=int(5e6), tb_log_name="second_run",
    #             callback=HParamCallback(), reset_num_timesteps=False)
    # model.learn(total_timesteps=int(5e6), tb_log_name="third_run",
    #             callback=HParamCallback(), reset_num_timesteps=False)
    # model.save('model_new')
    # run.finish()

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = vec_env.step(action)
        # vec_env.render()
        # VecEnv resets automatically
        # if done:
        #   # obs,_ = env.reset()
        if env.index == 124:
            print(env._get_state_str())
            print(env.fidelity)

    env.close()
