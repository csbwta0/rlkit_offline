import gym
import d4rl
import random
import numpy as np
import torch

from rlkit.core.logging import logger
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger,set_seed
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.cql_trainer import CQLTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

def experiment(variant):
    expl_env = gym.make(variant['env_id'])
    eval_env = gym.make(variant['env_id'])
    dataset = expl_env.get_dataset()
    
    obs_dim = expl_env.observation_space.low.size
    act_dim = expl_env.action_space.low.size
    
    M = variant['layer_size']
    
    seed = int(variant["seed"])
    set_seed(seed)
    
    qf1 = ConcatMlp(
        input_size = obs_dim+act_dim,
        hidden_sizes = [M,M,M],
        output_size = 1,
    )
    qf2 = ConcatMlp(
        input_size = obs_dim+act_dim,
        hidden_sizes = [M,M,M],
        output_size = 1,
    )
    
    target_qf1 = ConcatMlp(
        input_size=obs_dim + act_dim,
        output_size=1,
        hidden_sizes=[M,M,M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + act_dim,
        output_size=1,
        hidden_sizes=[M,M,M],
    )
    
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=act_dim,
        hidden_sizes=[M,M,M],
    )
    
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    print("Start Load Dataset")
    dataset_size = len(dataset['rewards'])
    for i in range(dataset_size):
        obs = dataset['observations'][i]
        next_obs = dataset['next_observations'][i]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        terminal = dataset['terminals'][i]
        replay_buffer.add_sample(obs,action,reward,terminal,next_obs)
    
    trainer = CQLTrainer(
        env = expl_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs'],
    )
    
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs'],
    )
    
    algorithm.to(ptu.device)
    algorithm.train()
    torch.save(trainer.policy.state_dict(),"./cql_model/policy.pth")
    torch.save(trainer.qf1.state_dict(),"./cql_model/qf1.pth")
    torch.save(trainer.qf2.state_dict(),"./cql_model/qf2.pth")
    torch.save(trainer.target_qf1.state_dict(),"./cql_model/target_qf1.pth")
    torch.save(trainer.target_qf2.state_dict(),"./cql_model/target_qf2.pth")
    
    
if __name__ == "__main__":
    variant = dict(
        env_id = "hopper-medium-replay-v2",
        algorithm = 'cql',
        version = 'test',
        seed = 0,
        layer_size = 256,
        replay_buffer_size = int(1e+6),
        algorithm_kwargs=dict(
            start_epoch=-1001,
            num_epochs=-1,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=3e-5,
            qf_lr=3e-4,
            reward_scale=1.0,
            use_automatic_entropy_tuning=True,
        ),
    )
    log_dir = (variant['algorithm']+'_'+variant['env_id'])
    setup_logger(exp_prefix=log_dir,variant=variant)
    ptu.set_gpu_mode(True,gpu_id=1)  # optionally set the GPU (default=False)
    experiment(variant)
    logger.writer.close()
    
    
    