"""Conservative Q Learning <ref: https://arxiv.org/pdf/2006.04779.pdf   >"""
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from rlkit.core.eval_util import create_stats_ordered_dict
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchTrainer

# CQL Lagrange


class CQLTrainer(TorchTrainer):
    def __init__(
        self,
        env: gym.Env,
        policy: nn.Module,
        qf1: nn.Module,
        qf2: nn.Module,
        target_qf1,
        target_qf2,

        discount=0.99,
        reward_scale=1.0,
        policy_lr=3e-5,
        qf_lr=3e-4,
        cql_alpha_lr=3e-4,  # varying alpha by dual gradient descent
        optimizer_class=optim.Adam,

        soft_target_tau=0.005,

        use_automatic_entropy_tuning=True,
        target_entropy=None,
        policy_eval_start=0,
        num_qs=2,

        # CQL
        temperature=1.0,
        cql_weight=5.0,

        num_random_actions=10,
        with_lagrange=False,
        lagrange_threshold=10.0,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_update_tau = soft_target_tau

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning

        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = - \
                    np.prod(self.env.action_space.shape).item()
            self.log_alpha = ptu.zeros(1, requires_grad=True)    # SAC中的alpha
            self.alpha_optimizer = optimizer_class([self.log_alpha], lr=1e-4)

        # CQL Lagrange
        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.lagrange_threshold = lagrange_threshold
            self.log_alpha_cql = ptu.zeros(1, requires_grad=True)  # CQL中的alpha
            self.alpha_cql_optimizer = optimizer_class(
                [self.log_alpha_cql], lr=cql_alpha_lr)

        self.polic_optimizer = optimizer_class(
            self.policy.parameters(), lr=policy_lr)
        self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=qf_lr)
        self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=qf_lr)

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.policy_eval_start = policy_eval_start

        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self._num_policy_steps = 1

        self.num_qs = num_qs

        self.temperature = temperature
        self.cql_weight = cql_weight

        self.num_random_action = num_random_actions

    def _get_q_values(self, obs: torch.Tensor, actions: torch.Tensor, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape//obs_shape)
        obs_tmp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(
            obs_shape*num_repeat, obs.shape[1])
        preds = network(obs_tmp.to(ptu.device), actions.to(ptu.device))
        preds = preds.view(obs_shape, num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_tmp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(
            obs.shape[0]*num_actions, obs.shape[1])
        new_dist = network(obs_tmp)
        new_obs_actions, new_log_prob = new_dist.rsample_and_logprob()

        return new_obs_actions, new_log_prob.detach().view(obs.shape[0], num_actions, 1)

    def train_from_torch(self, batch):
        """主要训练

        Args:
            batch:字典类型的数据
        """
        self._current_epoch += 1

        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        batch_size = len(rewards)

        dist = self.policy(obs)
        act, log_pi = dist.rsample_and_logprob()

        alpha = self.log_alpha.detach().exp()

        # Update Policy
        q1_act, q2_act = self.qf1(obs, act), self.qf2(obs, act)  # 用当前策略采样得到的动作
        policy_loss = (alpha*log_pi-torch.min(q1_act, q2_act)).mean()
        self.polic_optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)
        self.polic_optimizer.step()
        self._num_policy_update_steps += 1

        # Alpha Loss
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi.detach()+self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.detach().exp()

        else:
            alpha_loss = 0
            alpha = 1

        # TD Error
        q1, q2 = self.qf1(obs, actions), self.qf2(
            obs, actions)  # current q value
        with torch.no_grad():
            next_dist = self.policy(next_obs)
            next_actions, next_log_probs = next_dist.rsample_and_logprob()

            next_q = torch.min(self.target_qf1(next_obs, next_actions),
                               self.target_qf2(next_obs, next_actions)) - alpha * next_log_probs
            target_q = rewards*self.reward_scale + \
                self.discount*(1-terminals)*next_q

        critic1_loss = ((q1-target_q).pow(2)).mean()
        critic2_loss = ((q2-target_q).pow(2)).mean()

        # Conservative Loss
        # Paper Equation (30)
        # LogsumExp Q(s,a)
        random_actions = torch.FloatTensor(
            batch_size*self.num_random_action, actions.shape[-1]
        ).uniform_(self.env.action_space.low[0], self.env.action_space.high[0])

        curr_actions, curr_log_prob = self._get_policy_actions(
            obs, self.num_random_action, network=self.policy)
        new_actions, new_log_prob = self._get_policy_actions(
            next_obs, self.num_random_action, network=self.policy)

        q1_rand = self._get_q_values(obs, random_actions, network=self.qf1)
        q2_rand = self._get_q_values(obs, random_actions, network=self.qf2)

        q1_curr = self._get_q_values(obs, curr_actions, self.qf1)
        q2_curr = self._get_q_values(obs, curr_actions, self.qf2)

        q1_new = self._get_q_values(obs, new_actions, self.qf1)
        q2_new = self._get_q_values(obs, new_actions, self.qf2)

        # action_space.low[0]:-1.0    action_space.high[0]:1.0
        random_log_prob = np.log(
            0.5 ** random_actions.shape[-1])  # Q(s,a) - log_prob

        cat_q1 = torch.cat([q1_curr-curr_log_prob, q1_new -
                           new_log_prob, q1_rand-random_log_prob], 1)
        cat_q2 = torch.cat([q2_curr-curr_log_prob, q2_new -
                           new_log_prob, q2_rand-random_log_prob], 1)

        std_q1 = torch.std(cat_q1, dim=1)
        std_q2 = torch.std(cat_q2, dim=1)

        conservative_loss1 = torch.logsumexp(cat_q1/self.temperature, dim=1).mean(
        )*self.cql_weight*self.temperature-q1.mean()*self.cql_weight

        conservative_loss2 = torch.logsumexp(cat_q2/self.temperature, dim=1).mean(
        )*self.cql_weight*self.temperature-q2.mean()*self.cql_weight

        if self.with_lagrange:
            alpha_prime = torch.clamp(
                self.log_alpha_cql.exp(), min=0.0, max=1e+6)
            conservative_loss1 = alpha_prime * \
                (conservative_loss1-self.lagrange_threshold)
            conservative_loss2 = alpha_prime * \
                (conservative_loss2-self.lagrange_threshold)

            self.alpha_cql_optimizer.zero_grad()
            alpha_cql_loss = -(conservative_loss1 + conservative_loss2)*0.5
            alpha_cql_loss.backward(retain_graph=True)
            self.alpha_cql_optimizer.step()

        qf1_loss = critic1_loss + conservative_loss1
        qf2_loss = critic2_loss + conservative_loss2

        # Update Critic
        self._num_q_update_steps += 1
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        # Soft Updates
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_update_tau)
        ptu.soft_update_from_to(
            self.qf2, self.target_qf2, self.soft_update_tau)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))

            self.eval_statistics['Conservative QF1 Loss'] = np.mean(
                ptu.get_numpy(conservative_loss1))

            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Conservative QF2 Loss'] = np.mean(
                ptu.get_numpy(conservative_loss2))

            self.eval_statistics['Std QF1 values'] = np.mean(
                ptu.get_numpy(std_q1))
            self.eval_statistics['Std QF2 values'] = np.mean(
                ptu.get_numpy(std_q2))

            self.eval_statistics.update(create_stats_ordered_dict(
                'QF1 in-distribution values',
                ptu.get_numpy(q1_curr),
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                'QF2 in-distribution values',
                ptu.get_numpy(q2_curr),
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                'QF1 random values',
                ptu.get_numpy(q1_rand)
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                "QF2 random values",
                ptu.get_numpy(q2_rand),
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                "QF1 next_actions values",
                ptu.get_numpy(q1_new),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF2 next_actions values',
                ptu.get_numpy(q2_new),
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                'actions',
                ptu.get_numpy(actions),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'rewards',
                ptu.get_numpy(rewards),
            ))

            self.eval_statistics['Num Q Updates'] = self._num_q_update_steps
            self.eval_statistics['Num Policy Updates'] = self._num_policy_update_steps

            self.eval_statistics['Policy Loss'] = np.mean(
                ptu.get_numpy(policy_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                "Q1 Predictions", ptu.get_numpy(q1)))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions', ptu.get_numpy(q2)))

            self.eval_statistics.update(create_stats_ordered_dict(
                "Q Targets", ptu.get_numpy(target_q)))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['alpha'] = alpha.item()
                self.eval_statistics['alpha'] = alpha_loss.item()

            if self.with_lagrange:
                self.eval_statistics['alpha_cql'] = alpha_prime.item()
                self.eval_statistics['conservative_loss1'] = ptu.get_numpy(
                    conservative_loss1).mean()
                self.eval_statistics['conservative_loss2'] = ptu.get_numpy(
                    conservative_loss2).mean()

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]
        return base_list

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )


class Ensemble_CQLTrainer(TorchTrainer):
    def __init__(
        self,
        env: gym.Env,
        policy: nn.Module,
        qfs: list,
        target_qfs: list,

        discount=0.99,
        reward_scale=1.0,
        policy_lr=3e-5,
        qf_lr=3e-4,
        cql_alpha_lr=3e-4,  # varying alpha by dual gradient descent
        optimizer_class=optim.Adam,

        soft_target_tau=0.005,

        use_automatic_entropy_tuning=True,
        target_entropy=None,
        policy_eval_start=0,
        ensemble_size=3,

        # CQL
        temperature=1.0,
        cql_weight=5.0,

        num_random_actions=10,
        with_lagrange=False,
        lagrange_threshold=10.0,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qfs = qfs
        self.target_qfs = target_qfs

        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning

        if self.use_automatic_entropy_tuning:
            if target_entropy == None:
                self.target_entropy = - \
                    np.prod(self.env.action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = ptu.zeros(1, requires_grad=True)    # SAC中的alpha
            self.alpha_optimizer = optimizer_class([self.log_alpha], lr=1e-4)

        self.ensemble_size = ensemble_size
        self.temperature = temperature
        self.cql_weight = cql_weight
        self.num_random_actions = num_random_actions

        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.lagrange_threshold = lagrange_threshold
            self.log_alpha_cql = ptu.zeros(1, requires_grad=True)  # CQL中的alpha
            self.alpha_cql_optimizer = optimizer_class(
                [self.log_alpha_cql], lr=cql_alpha_lr)

        self.policy_optim = optimizer_class(
            self.policy.parameters(), lr=policy_lr)
        self.qfs_optim = []
        for i in range(self.ensemble_size):
            self.qfs_optim.append(optimizer_class(
                self.qfs[i].parameters(), lr=qf_lr))

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.policy_eval_start = policy_eval_start

        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self._num_policy_steps = 1

    def _get_q_values(self, obs, actions, network):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape//obs_shape)
        obs_tmp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(
            obs_shape*num_repeat, obs.shape[1])
        preds = network(obs_tmp.to(ptu.device), actions.to(ptu.device))
        preds = preds.view(obs_shape, num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network):
        obs_tmp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(
            obs.shape[0]*num_actions, obs.shape[1])
        new_dist = network(obs_tmp)
        new_obs_actions, new_log_prob = new_dist.rsample_and_logprob()

        return new_obs_actions, new_log_prob.detach().view(obs.shape[0], num_actions, 1)

    def train_from_torch(self, batch):
        self._current_epoch += 1
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        terminals = batch['terminals']
        rewards = batch['rewards']
        batch_size = len(rewards)

        act, log_pi = self.policy(obs).rsample_and_logprob()
        alpha = self.log_alpha.detach().exp()

        # Update Actor
        q_new_actions_all = []
        for i in range(self.ensemble_size):
            q_new_actions_all.append(self.qfs[i](obs, act))
        q_new_actions = torch.min(torch.hstack(
            q_new_actions_all), dim=1, keepdim=True).values

        policy_loss = (alpha * log_pi - q_new_actions).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self._num_policy_update_steps += 1

        # Update Alpha
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha*(log_pi.detach() +
                           self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.detach().exp()

        else:
            alpha_loss = 0
            alpha = 1

        # Update Critic

        # td error
        q_curr_action_all = []
        for i in range(self.ensemble_size):
            q_curr_action_all.append(self.qfs[i](obs, actions))

        with torch.no_grad():
            next_actions, next_log_pi = self.policy(
                next_obs).rsample_and_logprob()
            q_next_actions_all = []
            for i in range(self.ensemble_size):
                q_next_actions_all.append(
                    self.target_qfs[i](next_obs, next_actions))
            q_next_actions = torch.min(torch.hstack(
                q_next_actions_all), dim=1, keepdim=True).values
            next_q = q_next_actions - alpha * next_log_pi
            target_q = rewards * self.reward_scale + \
                self.discount * (1-terminals) * next_q

        random_actions = torch.FloatTensor(batch_size*self.num_random_actions, actions.shape[-1]).uniform_(
            self.env.action_space.low[0], self.env.action_space.high[0])
        curr_actions, curr_log_pi = self._get_policy_actions(
            obs, self.num_random_actions, network=self.policy)
        new_actions, new_log_pi = self._get_policy_actions(
            next_obs, self.num_random_actions, network=self.policy)

        critic_loss_all = []
        conservative_loss_all = []
        q_loss_all = []
        for i in range(self.ensemble_size):
            critic_loss = ((q_curr_action_all[i]-target_q).pow(2)).mean()
            q_rand = self._get_q_values(
                obs, random_actions, network=self.qfs[i])
            q_curr = self._get_q_values(obs, curr_actions, network=self.qfs[i])
            q_new = self._get_q_values(obs, new_actions, network=self.qfs[i])

            random_log_pi = np.log(0.5**random_actions.shape[-1])

            cat_q = torch.cat([q_curr-curr_log_pi, q_new -
                              new_log_pi, q_rand-random_log_pi], 1)

            conservative_loss = torch.logsumexp(cat_q/self.temperature, dim=1).mean(
            )*self.cql_weight*self.temperature-q_curr_action_all[i].mean()*self.cql_weight
            if self.with_lagrange:
                conservative_loss_all.append(
                    conservative_loss-self.lagrange_threshold)
            else:
                conservative_loss_all.append(conservative_loss)
            q_loss = critic_loss + conservative_loss
            q_loss_all.append(q_loss.detach())

            critic_loss_all.append(q_loss.detach())
            self.qfs_optim[i].zero_grad()
            q_loss.backward(retain_graph=True)
            self.qfs_optim[i].step()

        if self.with_lagrange:
            alpha_prime = torch.clamp(
                self.log_alpha_cql.exp(), min=0.0, max=1e+6)
            conservative_loss_all = torch.tensor(conservative_loss_all)
            alpha_cql_loss = alpha_prime * conservative_loss_all.mean()
            self.alpha_cql_optimizer.zero_grad()
            alpha_cql_loss.backward()
            self.alpha_cql_optimizer.step()

        # Soft Update
        for i in range(self.ensemble_size):
            ptu.soft_update_from_to(
                self.qfs[i], self.target_qfs[i], self.soft_target_tau)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            for i in range(self.ensemble_size):
                self.eval_statistics['Q{} Loss'.format(
                    i+1)] = np.mean(ptu.get_numpy(q_loss_all[i]))
                self.eval_statistics['Conservative{} Loss'.format(
                    i+1)] = np.mean(ptu.get_numpy(conservative_loss_all[i]))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['alpha'] = alpha.item()
                self.eval_statistics['alpha loss'] = alpha_loss.item()

            if self.with_lagrange:
                self.eval_statistics['alpha_cql'] = alpha_prime.item()

            self.eval_statistics['Policy Loss'] = np.mean(
                ptu.get_numpy(policy_loss))

        self._n_train_steps_total += 1

        uncertainty = torch.std(torch.hstack(q_curr_action_all), 1)
        uncertainty = ptu.get_numpy(uncertainty)
        return uncertainty

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = []
        for i in range(self.ensemble_size):
            base_list.append(self.qfs[i])
            base_list.append(self.target_qfs[i])

        base_list.append(self.policy)
        return base_list

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qfs=self.qfs,
            target_qfs=self.target_qfs
        )
