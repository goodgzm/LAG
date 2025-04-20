import sys
import os
#获取项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import logging
logging.basicConfig(level = logging.DEBUG)

class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype = torch.float32, device = torch.device('cpu'))
        self.use_prior = False

def _t2n(x):
    return x.detach().cpu().numpy()

num_agents = 1
render = True
ego_policy_index = 'latest'
episode_rewards = 0
ego_run_dir = "/data/gzm/LAG-master/scripts/results/SingleControl/1/heading/ppo/v1/wandb/latest-run/files"
experiment_name = ego_run_dir.split('/')[-4]

env = SingleControlEnv("1/heading")
env.seed(0)
args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device = torch.device("cuda"))
ego_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))


print("Start render")
obs = env.reset()
if render:
    env.render(mode = 'txt', filepath = f'{experiment_name}.txt.acmi')

ego_rnn_states = np.zeros((1, 1, 128), dtype = np.float32)
masks = np.ones((num_agents, 1))
ego_obs = obs[:num_agents, :]

while True:
    ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic = True)
    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)

    actions = ego_actions
    obs, rewards, dones, infos = env.step(actions)
    episode_rewards += rewards

    if render:
        env.render(mode = 'txt', filepath = f'{experiment_name}.txt.acmi')

    if dones.all():
        print(infos)
        break
    
    bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
    print(f"step : {env.current_step}, bloods:{bloods}")
    ego_obs = obs[:num_agents:]

print(episode_rewards)


