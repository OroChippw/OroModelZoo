# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def smooth(a,winsize):
    """
    Smoothing with edge processing.
    Input:
        a:原始数据，NumPy 1-D array containing the data to be smoothed,必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化 
        winsize: smoothing window size needs, which must be odd number,as in the original MATLAB implementation
    Output:
        
    """
    out0 = np.convolve(a,np.ones(winsize,dtype=int),'valid')/winsize
    r = np.arange(1,winsize-1,2)
    start = np.cumsum(a[:winsize-1])[::2]/r
    stop = (np.cumsum(a[:-winsize:-1])[::2]/r)[::-1]
    return np.concatenate(( start , out0, stop ))

def k_armed_bandit_once(value_list , epsilon , nStep , updateAlgo='sample_average' , alpha=0 , stationary=True):
    '''
        Func:
            One run of k-armed bandit simulation
        Args:
            value_list : reward for each candition actions
            epsilon : greedy factor ϵ for epsilon-greedy algorithm
            nStep : the number of steps for simulation
            updateAlgo : the algorithm for updating action-value
            alpha : step-size in case of 'exp_decaying'
        Return:
            step_action : action series for each step in one run
            action_counter : The number of being selected for action[k]
            action_value_list : reward sample average up to t-1 for action[k]
            step_reward : reward series for each step in one run
            optimal_ratio : Ration of optimal action being selected over time
    '''
    action_num = len(value_list)
    action_value_list = np.zeros(action_num) # estimation of action value.
    action_counter = np.zeros(action_num , dtype='int') # record the number of Action_#k being selected
    step_action = np.zeros(nStep + 1 , dtype='int') # record the adopted action in each time step
    step_reward = np.zeros(nStep + 1)
    
    if not stationary:
        value_list = np.ones(action_num) / action_num
    
    optimal_counter = 0 # count the number of time steps in which the optimal action is selected
    optimal_ratio = np.zeros(nStep + 1 , dtype="float")
    
    for t in range(1 , nStep + 1):
        # ground-truth optimal action, with the largest qstar value
        # 对于mom-stationary env，optimal_action经常改变所以移到循环内部
        optimal_action = np.argmax(value_list) 
        probability = np.random.uniform(0,1) # select action
        if probability < epsilon: # random selection for exploring
            step_action[t] = np.random.choice(np.arange(action_num))
        else:
            # 使用np.random.permutation的目的是为了确保在进行贪心选择时能够均匀的在并列的最大动作中进行随机选择，已增加算法的探索性            
            p = np.random.permutation(action_num)
            step_action[t] = p[np.argmax(action_value_list[p])]
    
        action_counter[step_action[t]] += 1
        
        # Reward,使用randn()引入一些随机性（服从标准正态分布），模拟真实世界中奖励的随机性和不确定性
        step_reward[t] = np.random.randn() + value_list[step_action[t]]
        
        if updateAlgo == 'sample_average':
            # Sample Average,进行了加权形式的改进
            # action_value_list[step_action[t]] = (action_value_list[step_action[t]] * (action_counter[step_action[t]] - 1) + step_reward[t]) / action_counter[step_action[t]]
            '''
            基于增量式实现，递推的通用形式有NewEstimate <-- OldEstimate + StepSize*[Target - OldEstimate]
            [Target - OldEstimate]表示当前估计值与目标值的差距或者说估计误差项
            基于以上递推关系，我们可以重新改写行动价值估计的代码
                Q[a[t]] = Q[a[t]] + (r[t]-Q[a[t]])/aNum[a[t]]   
            上面推导时只考虑一个行动。但是实际代码实现中需要对K个行动进行跟踪，所以代码比上面的递推关系式要显得更复杂一些。每次只针对当前时刻所采取的行动更新其行动价值估计，Q值以及行动被选择的次数都需要针对K个行动分别存储更新
            '''
            action_value_list[step_action[t]] = (action_value_list[step_action[t]] + (step_reward[t] - action_value_list[step_action[t]])) / action_counter[step_action[t]]
        elif updateAlgo == 'exp_decaying':
            action_value_list[step_action[t]] = action_value_list[step_action[t]] + (step_reward[t] - action_value_list[step_action[t]]) * alpha

        # Optimal Action Ratio Tracking,计算前t个时间步中选择最优动作的比例
        if step_action[t] == optimal_action:
            optimal_counter += 1
        optimal_ratio[t] = optimal_counter / t

        #  Random walk of qstar simulating non-stationary environment
        if not stationary:
            # 在每一步之后，每个行动的qstar叠加一个随机值，随机值从从零均值，标准偏差为0.01的正态分布中抽取
            value_list = value_list + np.random.randn(action_num) * 0.01 # 标准差为0.01
            
    # print(f"Action[{len(step_action)}] : {step_action}")
    # print(f"Action Counter[{len(action_counter)}] : {action_counter}")
    # print(f"Action-Value[{len(action_value_list)}] : {action_value_list}")
    # print(f"Reward[{len(step_reward)}] : {step_reward}")
    # print(f"Optimal Ratio[{len(optimal_ratio)}] : {optimal_ratio}")
    
    return step_action , action_counter , action_value_list , step_reward , optimal_ratio

def k_armed_bandit_one_run(qstar,epsilon,nStep,QUpdtAlgo='sample_average',alpha=0, stationary=True):
    """
    One run of K-armed bandit simulation.
    Input:
        qstar:     Mean reward for each candition actions
        epsilon:   Epsilon value for epsilon-greedy algorithm
        nStep;     The number of steps for simulation
        QUpdtAlgo: The algorithm for updating Q value--'sample_average','exp_decaying'
        alpha:     step-size in case of 'exp_decaying'
    Output:
        a[t]: action series for each step in one run
        r[t]: reward series for each step in one run
        Q[k]: reward sample average up to t-1 for action[k]
        aNum[k]: The number of being selected for action[k]
        optRatio[t]: Ration of optimal action being selected over tim
    """

    K     = len(qstar)
    Q     = np.zeros(K)
    a     = np.zeros(nStep+1,dtype='int') # Item#0 for initialization
    aNum  = np.zeros(K,dtype='int')       # Record the number of action#k being selected

    r     = np.zeros(nStep+1)             # Item#0 for initialization

    if stationary == False:
        qstar = np.ones(K)/K                 # qstart initialized to 1/K for all K actions    

    optCnt   = 0
    optRatio = np.zeros(nStep+1,dtype='float') # Item#0 for initialization

    for t in range(1,nStep+1):

        #0. For non-stationary environment, optAct also changes over time.Hence, move to inside the loop.
        optAct   = np.argmax(qstar)
        #1. action selection
        tmp = np.random.uniform(0,1)
        #print(tmp)
        if tmp < epsilon: # random selection
            a[t] = np.random.choice(np.arange(K))
            #print('random selection: a[{0}] = {1}'.format(t,a[t]))
        else:             # greedy selection
            #选择Q值最大的那个，当多个Q值并列第一时，从中任选一个--但是如何判断有多个并列第一的呢？
            #对Q进行random permutation处理后再找最大值可以等价地解决这个问题
            p = np.random.permutation(K)
            a[t] = p[np.argmax(Q[p])]
            #print('greedy selection: a[{0}] = {1}'.format(t,a[t]))

        aNum[a[t]] = aNum[a[t]] + 1

        #2. reward: draw from the pre-defined probability distribution    
        r[t] = np.random.randn() + qstar[a[t]]        

        #3.Update Q of the selected action - #2.4 Incremental Implementation
        # Q[a[t]] = (Q[a[t]]*(aNum[a[t]]-1) + r[t])/aNum[a[t]]    
        if QUpdtAlgo == 'sample_average':
            Q[a[t]] = Q[a[t]] + (r[t]-Q[a[t]])/aNum[a[t]]    
        elif QUpdtAlgo == 'exp_decaying':
            Q[a[t]] = Q[a[t]] + (r[t]-Q[a[t]])*alpha
        
        #4. Optimal Action Ratio tracking
        #print(a[t], optAct)
        if a[t] == optAct:
            optCnt = optCnt + 1
        optRatio[t] = optCnt/t

        #5. Random walk of qstar simulating non-stationary environment
        # Take independent random walks (say by adding a normally distributed increment with mean 0
        # and standard deviation 0.01 to all the q⇤(a) on each step).   
        if stationary == False:        
            qstar = qstar + np.random.randn(K)*0.01 # Standard Deviation = 0.01
            #print('t={0}, qstar={1}, sum={2}'.format(t,qstar,np.sum(qstar)))
        
    return a,aNum,r,Q,optRatio

def main():
    K = 10 # armed num
    epsilon = 0.1 # greedy threshold
    step = 10000
    value_list = np.random.randn(K) # value of each actions
    
    print(f"Value = {value_list}")
    print(f"Optimal Action is Action#{np.argmax(value_list)}")
    
    # Once K-armed bandit
    action , action_counter , action_values , reward , optimal_ratio = k_armed_bandit_once(value_list , epsilon , step , updateAlgo="sample_average")
    
    fig , ax = plt.subplots()
    
    # Histogram of action sequence
    plt.hist(action) 
    plt.title('Histogram of action sequence')
    
    # Optimal selection ratio along the time
    # fig , ax = plt.subplots()
    # plt.plot(optimal_ratio)
    # plt.title('Optimal selection ratio along the time')
    
    # One trial of epsilon-greedy k-armed badit game
    # Difference between estimation and ground-truth of action value
    fig , ax = plt.subplots(1,2,figsize=[12,6])
    ax[0].scatter(action_values,value_list)
    ax[0].grid()
    ax[0].set_title('One trial of epsilon-greedy k-armed badit game')
    ax[1].plot(action_values - value_list)
    ax[1].grid()
    ax[1].set_title('Difference between estimation and ground-truth of action value')
    ax[1].set_ylim(-1.5,2.0)
    
    # Number of actions vs qstar
    # 看各种行动被选择的次数以及它们的价值之间的对比关系，各action被选择的次数应该遵循马太效应，即最后应该集中在最高q值的action或最高的几个actions上
    fig, ax2_1 = plt.subplots()
    ax2_1.scatter(np.arange(K),action_counter)
    ax2_1.set_ylabel("Number of action")
    
    ax2_2 = ax2_1.twinx()
    ax2_2.plot(np.arange(K) , value_list, label='value', color='blue', marker='s')
    ax2_2.set_ylabel("value")
    ax2_2.set_ylim(np.min(value_list) - 0.5, np.max(value_list) + 0.5)
    ax2_2.legend(loc='upper left')
    # print(action_counter)
    # print(value_list)

    plt.show()
    
    return 

def main2():
    # 平稳环境下对比两种Q值估计方法的表现
    nStep  = 1000
    nRun   = 1000
    K      = 10
    alpha  = 0.1
    r_smpaver = np.zeros((nRun,nStep+1))
    optRatio_smpaver  = np.zeros((nRun,nStep+1))

    r_exp = np.zeros((nRun,nStep+1))
    optRatio_exp  = np.zeros((nRun,nStep+1))

    for run in range(nRun):
        print('.',end='')
        if run%100==99:        
            print('run = ',run+1)

        qstar   = np.random.randn(K) 
        a_smpaver,aNum_smpaver,r_smpaver[run,:],Q,optRatio_smpaver[run,:] = k_armed_bandit_one_run(qstar,0.1,nStep)
        a_exp,aNum_exp,r_exp[run,:],Q,optRatio_exp[run,:] = k_armed_bandit_one_run(qstar,0.1,nStep,'exp_decaying',alpha)

    rEnsembleMean_smpaver = np.mean(r_smpaver,axis=0)
    optRatioEnsembleMean_smpaver = np.mean(optRatio_smpaver,axis=0)
    
    rEnsembleMean_exp = np.mean(r_exp,axis=0)
    optRatioEnsembleMean_exp = np.mean(optRatio_exp,axis=0)
    
    fig,ax = plt.subplots(1,2,figsize=(15,4))
    ax[0].plot(smooth(rEnsembleMean_smpaver,5))
    ax[0].plot(smooth(rEnsembleMean_exp,5))
    ax[1].plot(smooth(optRatioEnsembleMean_smpaver,5))
    ax[1].plot(smooth(optRatioEnsembleMean_exp,5))
    ax[0].legend(['sample average method','exponential decaying'])
    ax[1].legend(['sample average method','exponential decaying'])
    ax[0].set_title('ensemble mean reward')
    ax[1].set_title('ensemble mean optimal ratio')
    
    plt.show()
    
def main3():
    # 非平稳环境下对比两种Q值估计方法的表现
    nStep  = 40000
    nRun   = 1000
    K      = 10
    alpha  = 0.1
    r_smpaver = np.zeros((nRun,nStep+1))
    optRatio_smpaver  = np.zeros((nRun,nStep+1))
    
    r_exp = np.zeros((nRun,nStep+1))
    optRatio_exp  = np.zeros((nRun,nStep+1))
    
    for run in range(nRun):
        print('.',end='')
        if run%100==99:        
            print('run = ',run+1)
        
        qstar   = np.random.randn(K) 
        a_smpaver,aNum_smpaver,r_smpaver[run,:],Q,optRatio_smpaver[run,:] = k_armed_bandit_one_run(qstar,0.1,nStep,'sample_average',alpha,False)
        a_exp,aNum_exp,r_exp[run,:],Q,optRatio_exp[run,:] = k_armed_bandit_one_run(qstar,0.1,nStep,'exp_decaying',alpha,False)
    
    rEnsembleMean_smpaver = np.mean(r_smpaver,axis=0)
    optRatioEnsembleMean_smpaver = np.mean(optRatio_smpaver,axis=0)
    
    rEnsembleMean_exp = np.mean(r_exp,axis=0)
    optRatioEnsembleMean_exp = np.mean(optRatio_exp,axis=0)
    
    fig,ax = plt.subplots(1,2,figsize=(15,4))
    ax[0].plot(smooth(rEnsembleMean_smpaver,5))
    ax[0].plot(smooth(rEnsembleMean_exp,5))
    ax[1].plot(smooth(optRatioEnsembleMean_smpaver,5))
    ax[1].plot(smooth(optRatioEnsembleMean_exp,5))
    ax[0].legend(['sample average method','exponential decaying'])
    ax[1].legend(['sample average method','exponential decaying'])
    ax[0].set_title('ensemble mean reward')
    ax[1].set_title('ensemble mean optimal ratio')
    
    plt.show()
    
if __name__ == '__main__':
    
    # main()
    
    # 平稳环境下对比两种Q值估计方法的表现
    # main2()
    
    # 非平稳环境下对比两种Q值估计方法的表现
    # 在非平稳环境下，''exponential recency-weighted average''方法的优势明显
    main3()
    