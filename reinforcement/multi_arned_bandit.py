# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit():
    """ 伯努利多臂老虎机,输入K表示拉杆个数 """
    def __init__(self , K) -> None:
        self.K = K
        self.probs = np.random.uniform(size=K)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        pass

    def step(self , k):
        """ 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未获奖）"""
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

class Solver():
    """ 多臂老虎机算法基本框架 """
    def __init__(self , bandit) -> None:
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K) # 每根拉杆的尝试次数
        self.regret = 0 # 当前步的累积loss
        self.actions = [] # 记录每一步的动作列表
        self.regrets = [] # 记录每一步的累积loss
    
    def update_regret(self , k):
        self.regret += (self.bandit.best_prob - self.bandit.probs[k])
        self.regrets.append(self.regret)
    
    def run_one_step(self):
        raise NotImplementedError
    
    def run(self , num_steps):
        # num_steps为运行的总次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)
            

class EpsilonGreedy(Solver):
    def __init__(self, bandit , epsilon=0.01 , init_prob=1.0) -> None:
        super(EpsilonGreedy , self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K) # 初始化拉动所有拉杆的期望奖励估值
    
    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0 , self.bandit.K) # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates) # 选择期望估值最大的拉杆
        
        r = self.bandit.step(k)
        '''
            self.counts[k] + 1防止分母为零的情况
        '''
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        
        return k
    
class DecayingEpsilonGreedy(Solver):
    """ epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类 """
    def __init__(self , bandit , init_prob=1.0) -> None:
        super(DecayingEpsilonGreedy , self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K) # 初始化拉动所有拉杆的期望奖励估值
        self.total_count = 0
    
    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < (1 / self.total_count):
            k = np.random.randint(0 , self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        
        return k

def plot_results(solvers , solver_names):
    '''
        生成累积loss随时间变化的图像，输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称
    '''
    for idx , solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list , solver.regrets , label=solver_names[idx])
    
    plt.xlabel('Time Step')
    plt.xlabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()
    
    
def main_BernoulliBandit():
    np.random.seed(0)
    K = 10
    bandit_arm = BernoulliBandit(K)
    print("[INFO] 随机生成了一个%d臂伯努利老虎机" % K)
    print("[INFO] 获奖概率最大的拉杆为%d号,其获奖概率为%.4f" % (bandit_arm.best_idx, bandit_arm.best_prob))
    epsilon_greedy_solver = EpsilonGreedy(bandit_arm, epsilon=0.01)
    epsilon_greedy_solver.run(5000)
    print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])
    
    return 

def main_diff_epsilon_BernoulliBandit():
    '''
        无论epsilon取值多少，累积懊悔都是线性增长的。在这个例子中，随着epsilon的增大，累积懊悔增长的速率也会增大
    '''
    np.random.seed(0)
    epsilons = [1e-4 , 0.01 , 0.1 , 0.25 , 0.5]
    K = 10
    bandit_arm = BernoulliBandit(K)
    epsilon_greedy_solver_list = [
        EpsilonGreedy(bandit_arm, epsilon=e) for e in epsilons
    ]
    epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
    for solver in epsilon_greedy_solver_list:
        solver.run(5000)

    plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)
    
    return

def main_decaying_epsilon_BernoulliBandit():
    '''
        随时间做反比例衰减的epsilon-贪婪算法能够使累积懊悔与时间步的关系变成次线性（sublinear）的，这明显优于固定epsilon值的epsilon-贪婪算法
    '''
    np.random.seed(1)
    K = 10
    bandit_arm = BernoulliBandit(K)
    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_arm)
    decaying_epsilon_greedy_solver.run(5000)
    print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
    plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])
    
    return 

if __name__ == '__main__':    
    # main_BernoulliBandit()
    
    # main_diff_epsilon_BernoulliBandit()
    # https://hrl.boyuai.com/chapter/1/%E5%A4%9A%E8%87%82%E8%80%81%E8%99%8E%E6%9C%BA#21-%E7%AE%80%E4%BB%8B
    
    main_decaying_epsilon_BernoulliBandit()