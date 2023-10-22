import gymnasium as gym
from SFQenv import SFQ
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env

# env = gym.make("LunarLander-v2")
env = SFQ()
#vec_env = make_vec_env(SFQ,n_envs=3)
#model = PPO("MlpPolicy", env, verbose=1,batch_size=6000,ent_coef=0.00,learning_rate=5e-4)
model = PPO.load("model.zip", env=env, ent_coef=0.00, batch_size=6000, learning_rate=4e-4)
model.learn(total_timesteps=int(15e6))
model.save('model')
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
        print(env.true_fidelity)

env.close()
