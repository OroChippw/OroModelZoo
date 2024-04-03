# -*- coding: utf-8 -*-
# https://blog.csdn.net/Er_Studying_Bai/article/details/128462002

import matplotlib
import gymnasium as gym
import matplotlib.pyplot as plt

import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from collections import namedtuple , deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
    a named tuple representing a single transition in our env
    (state , action) -> ('next_state' , 'reward')
'''
Transition = namedtuple(
    'Transition', ('state' , 'action'  , 'reward' , 'next_state'  , 'done'))

class ReplayMemory(object):
    '''
        A cyclic buffer of bounded size that holds the transitions observed recently
    '''
    def __init__(self , capacity) -> None:
        self.memory = deque([] , maxlen=capacity)
    
    def push(self , *args):
        self.memory.append(Transition(*args))
    
    def sample(self , batch_size):
        # 每次DQN更新的时候，随机抽取一些之前的经历进行学习，随机抽取的原因是打乱经历之间的相关性
        batch_data = random.sample(self.memory , batch_size)
        state , action , reward , next_state , done = zip(*batch_data)
        return state , action , reward , next_state , done
    
    def __len__(self):
        return len(self.memory)

class DeepQNetwork(nn.Module):
    """
        the network is trying to predict the expected return of taking each action given the current input.
    """
    def __init__(self , n_observations , n_actions) -> None:
        super(DeepQNetwork , self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_observations , 128) , 
            nn.ReLU() , 
            nn.Linear(128 , n_actions)
        )

    def forward(self , x):
        result_ = self.model(x)
        
        return result_

class Agent(object):
    def __init__(self , observation_dim , action_dim , gamma , lr , epslion , target_update) -> None:
        self.action_dim = action_dim
        # policy_net,用于选择动作。它根据当前状态预测每个动作的价值，并选择具有最高价值的动作执行
        self.policy_net = DeepQNetwork(observation_dim , action_dim).to(device)
        # target_net用于计算目标Q值。在训练过程中，我们使用目标网络来计算当前状态的目标Q值
        self.target_net = DeepQNetwork(observation_dim , action_dim).to(device)    
        self.gamma = gamma
        self.lr = lr
        self.epslion = epslion
        self.target_update = target_update
        self.count = 0
        
        self.optimizer = optim.Adam(params=self.policy_net.parameters() , lr=lr)
        self.loss = nn.MSELoss()
    
    def take_action(self , state):
        if np.random.uniform(0,1) < (1 - self.epslion):
            state = torch.tensor(state , dtype=torch.float).to(device)
            action = torch.argmax(self.policy_net(state)).item()
        else:
            action = np.random.choice(self.action_dim)
        
        return action
    
    def update(self , transition_dict):
        states = transition_dict.state
        actions = np.expand_dims(transition_dict.action ,axis=-1)
        rewards = np.expand_dims(transition_dict.reward , axis=-1)
        next_states = transition_dict.next_state
        dones = np.expand_dims(transition_dict.done , axis=-1)
        
        states = torch.tensor(states , dtype=torch.float).to(device)
        actions = torch.tensor(actions , dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards , dtype=torch.float).to(device)
        next_states = torch.tensor(next_states , dtype=torch.float).to(device)
        dones = torch.tensor(dones , dtype=torch.float).to(device)
        
        predict_q_values = self.policy_net(states).gather(1,actions)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
            
        loss = self.loss(predict_q_values , q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        self.optimizer.step()
        
        if self.count % self.target_update == 0:
            # Copt model parameters
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.count += 1

def run_episode(env , agent , memory , batch_size):
    state , info = env.reset()
    reward_total = 0
    while True:
        action = agent.take_action(state)
        next_state , reward , done , _  , _ = env.step(action) 
        memory.push(state , action , reward , next_state , done)
        reward_total += reward
        if len(memory) > batch_size:
            state_batch , action_batch , reward_batch , next_state_batch , done_batch = memory.sample(batch_size)
            data = Transition(state_batch , action_batch , reward_batch , next_state_batch , done_batch) 
            agent.update(data)
        
        state = next_state
        if done:
            break
    
    return reward_total

def episode_evaluate(env , agent , render=False):
    reward_list = []
    for idx in range(5):
        state ,info = env.reset()
        reward_episode = 0
        while True:
            action = agent.take_action(state)
            next_state , reward , done , _  , _ = env.step(action)
            reward_episode += reward
            state = next_state
            if done:
                break
            if render:
                env.render()
        reward_list.append(reward_episode)
    
    return np.mean(reward_list).item()
    
def test(env , agent , delay_time):
    state ,info = env.reset()
    reward_episode = 0
    while True:
        action = agent.take_action(state)
        next_state , reward , done , _ , _ = env.step(action)
        reward_episode += reward
        state = next_state
        if done:
            break
        env.render()
        time.sleep(delay_time)

def main():
    '''
        CartPole运行结束条件：
            （1）杆子角度超出[-12,12]度
            （2）小车的位置超出[-2.4,2.4]，小车的中心到达显示屏边缘
            （3）小车移动步数超过200（v1为500）
            小车每走一步奖励就会+1，所以在v0版本环境中，小车一次episode的最大奖励为200。
    '''
    env = gym.make("CartPole-v0" , render_mode="human")
    env_name = "CartPole-v0"
    observation_n , action_n = env.observation_space.shape[0] , env.action_space.n
    print(f"[INFO] observation_n : f{observation_n} , action_n : {action_n}")
    agent = Agent(observation_n ,action_n , gamma=0.98 , lr=1e-3 , epslion=0.01 , target_update=10)
    
    memory = ReplayMemory(1000)
    
    batch_size = 64
    num_episodes = 200
    reward_list = []
    
    print(f"[INFO] Training model ...")
    for idx in range(10):
        with tqdm(total=int(num_episodes / 10) , desc="Iteration %d" % idx) as pbar:
            for episode in range(int(num_episodes / 10)):
                reward_episode = run_episode(env ,agent , memory , batch_size)
                reward_list.append(reward_episode)
                
                if (episode + 1) % 10 == 0:
                    test_reward = episode_evaluate(env , agent , True)
                    pbar.set_postfix({
                        'episode' : '%d' % (num_episodes / 10 * idx + episode * idx),
                        'return' : '%.3f' % (test_reward)
                    })
                pbar.update(1)
    
    # test(env , agent , 0.5)
    episode_list = list(range(len(reward_list)))
    plt.plot(episode_list , reward_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'Double DQN on {env_name}')
    plt.show()
                                
    return None

if __name__ == '__main__':
    main()
        


