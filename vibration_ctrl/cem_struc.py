import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch import jit
from torch import nn, optim
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import scipy.io as sio
import time

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

data = sio.loadmat("data_lin.mat")
Ad = torch.from_numpy(data['Ad'].astype(np.float32)).transpose(0,1)
Bd = torch.from_numpy(data['Bd'].astype(np.float32)).transpose(0,1)
Cd = torch.from_numpy(data['Cd'].astype(np.float32)).transpose(0,1)
Dd = torch.from_numpy(data['Dd'].astype(np.float32)).transpose(0,1)

action_true = - torch.from_numpy(data['ctrlForce'][:,:].astype(np.float32))
obs_true = torch.from_numpy(data['y'][:,:].astype(np.float32))
state_true = torch.from_numpy(data['X'][:,:].astype(np.float32))

class FuncMinGTEnv():
    def __init__(self, initial_time, planning_horizon, batch_size, input_size, batched_func):
        self.a_size = input_size
        self.func = batched_func
        self.state = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.reset_state(batch_size)
        self.initial_time = initial_time
        self.planning_horizon = planning_horizon

    def reset_state(self, batch_size):
        self.B = batch_size

    def rollout(self, actions):
        # Uncoditional action sequence rollout
        # TxBxA
        T = actions.size(0)
        total_r = torch.zeros(self.B, requires_grad=True, device=actions.device)
        _, r, done = self.step(actions)
        total_r = total_r + r
            #if(done):
            #    break
        return total_r

    def step(self, action):
        obs = obs_true[self.initial_time:self.initial_time+self.planning_horizon]
        obs = obs.view(self.planning_horizon,1,10).repeat(1,1000,1)
        self.state = self.sim(self.state, action)
        o = self.calc_obs(self.state)
        r = self.calc_reward(self.state, action, obs)
        # always done after first step
        return o, r, True

    def sim(self, state, action):
        s_prev = state_true[self.initial_time]
        s_prev = s_prev.repeat(1,1000,1)
        s = s_prev
        for i in range(self.planning_horizon-1):
            s_next = torch.matmul(s_prev, Ad)
            s_next = s_next + torch.matmul(action[i], Bd)
            s = torch.cat([s, s_next])
            s_prev = s_next
        return s

    def calc_obs(self, state):
        return None

    def calc_reward(self, state, action, obs):
        return -self.func(self.initial_time, self.planning_horizon, state, action, obs)


def get_test_energy2d_env(initial_time, planning_horizon, batch_size):
    return FuncMinGTEnv(initial_time, planning_horizon, batch_size, 10, test_energy2d)

def test_energy2d(initial_time, planning_horizon, state_batch, action_batch, obs_batch):
    action_true_batch = action_true[initial_time:initial_time+planning_horizon].view(planning_horizon,1,10).repeat(1,1000,1)
    opt_point = state_true[initial_time:initial_time+planning_horizon].view(planning_horizon,1,20).repeat(1,1000,1)
    return (torch.matmul(state_batch,Cd)**2 + action_batch**2).sum(-1).sum(0)


class CEM():  # jit.ScriptModule):
    def __init__(self, planning_horizon, opt_iters, samples, top_samples, env, device):
        super().__init__()
        self.set_env(env)
        self.H = planning_horizon
        self.opt_iters = opt_iters
        self.K, self.top_K = samples, top_samples
        self.device = device

    def set_env(self, env):
        self.env = env
        if self.env is not None:
            self.a_size = env.a_size

    # @jit.script_method
    def forward(self, batch_size, return_plan=False, return_plan_each_iter=False):
        # Here batch is strictly if multiple CEMs should be performed!
        B = batch_size

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        a_mu = torch.zeros(self.H, 1, self.a_size, device=self.device)
        a_std = 1*torch.ones(self.H, 1, self.a_size, device=self.device)
        plan_each_iter = []
        for _ in range(self.opt_iters):
            self.env.reset_state(self.K)
            # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
            # Sample actions (T x (B*K) x A)
            actions = (a_mu + a_std * torch.randn(self.H, self.K, self.a_size, device=self.device)).view(self.H, self.K, self.a_size)
            # Returns (B*K)
            returns = self.env.rollout(actions)

            # Re-fit belief to the K best action sequences
            _, topk = returns.reshape(B, self.K).topk(self.top_K, dim=1, largest=True, sorted=False)
            topk += self.K * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)
            best_actions = actions[:, topk.view(-1)].reshape(self.H, B, self.top_K, self.a_size)
            # Update belief with new means and standard deviations
            a_mu = best_actions.mean(dim=2, keepdim=False)
            a_std = best_actions.std(dim=2, unbiased=False, keepdim=False)

            if return_plan_each_iter:
                _, topk = returns.reshape(B, self.K).topk(1, dim=1, largest=True, sorted=False)
                best_plan = actions[:, topk[0]].reshape(self.H, B, self.a_size).detach()
                plan_each_iter.append(best_plan.data.clone())

        if return_plan_each_iter:
            return plan_each_iter
        if return_plan:
            return a_mu.squeeze(dim=2)
        else:
            # Return first action mean µ_t
            return a_mu[0]




K = 1000
tK = 50
H = 20
dlen = 140
it = 100
initial_time=0
obs_true = obs_true

tStart = time.time()
for i in range(dlen):
    if i == 0:
        initial_time=0
        t_env = get_test_energy2d_env(initial_time, H, K)
        planner = CEM(H, it, K, tK, t_env, device=torch.device('cpu'))
        action = planner.forward(1)
        actions = action.data
        torch.cuda.empty_cache()
    else:
        initial_time=i
        t_env = get_test_energy2d_env(initial_time, H, K)
        planner = CEM(H, it, K, tK, t_env, device=torch.device('cpu'))
        action = planner.forward(1)
        actions=torch.cat([actions, action.data])
        torch.cuda.empty_cache()
tEnd = time.time()-tStart
print('ILfO Execution time {}'.format(tEnd))

plt.rc('font', family='Times New Roman')
plt.rcParams["mathtext.fontset"] = "cm"

plt.figure(figsize=(80,32))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.plot(action_true[0:dlen,i], label='true inputs', lw=12, color='black')
    plt.plot(actions[:,i], label='estimated inputs', lw=12, linestyle='--', color='red')
    plt.ylim([-2.5,2.5])
    plt.xticks(fontsize=80)
    plt.yticks(fontsize=80)
    if i == 0:
        plt.legend(fontsize=70, loc='upper left')
    if i == 0 or i == 5:
        plt.ylabel('Input force [N]', fontsize=80)
    if i in [5,6,7,8,9]:
        plt.xlabel('Time [s]', fontsize=80)

plt.figure()
plt.plot(actions[:,0])
plt.plot(action_true[0:dlen,0])

error = 0
for i in range(10):
    error += np.sqrt(mean_squared_error(actions[:,i], action_true[0:dlen,i]))
error /= 10
print('ILfO MSE: {}'.format(error))


import seaborn as sns
from matplotlib.pyplot import rc as rc
sns.set_style("whitegrid")
rc('font', family='serif')
rc('text', usetex=True)
plt.figure(figsize=(50,32))
for i in range(10):
    plt.subplot(5,2,i+1)
    plt.plot(action_true[:100,i], label='Ground-truth', color='black', lw = 10)
    plt.plot(actions[:100,i], label='ILfO', linestyle='--', color='#BB5566', lw = 10)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.ylim([-0.2,0.15])
    if i == 0:
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0,1]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],\
                   loc='upper left', bbox_to_anchor=(0.0, 1.5), ncol=2, fontsize=50, framealpha=1)
    if i in [8,9]:
        plt.xlabel('Time [s]', fontsize=50)
    if i in range(9):
        plt.ylabel(f'$u_{i+1}$ [N]', fontsize=50)
    if i == 9:
        plt.ylabel('$u_{10}$ [N]', fontsize=50)
