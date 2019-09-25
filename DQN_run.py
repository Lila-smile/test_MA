from Main_ECM import Env_battery  # import environment ############
from DQN import DeepQNetwork
import pandas as pd
from mat4py import loadmat
import numpy as np
import matplotlib.pyplot as plt
# def run_env(observation, time):
#
#     #for episode in range(2000):
#         ## initial observation
#         #observation = np.array(Env_battery.reset())
#         while True:
#             #for time in range(len(final_data) + 1) :
#                 # RL choose action based on observation
#                 action = RL.choose_action(observation)
#
#                 # RL take action and get next observation and reward
#                 observation_, reward, done = Env_battery_update.step(action, time)  #####time stamp of action
#
#                 RL.store_transition(observation, action, reward, observation_)
#
#                 RL.learn()
#
#                 # swap observation
#                 observation = observation_
#
#                 # break while loop when end of this episode
#                 if done:
#                     break
#     # end of game
#     # print('game over')
#     # env.destroy()


if __name__ == "__main__":
    # set environment

    #data = loadmat('/home/cuihan/PycharmProjects/masterthesisproject/testdata/2014-03-21_MEA_0046_80300894.mat')  # print(type(data))
    #current = data['BATT_CURRENT'][1:86378:100]
    #voltage = data['BATT_V_TOTAL'][1:86378:100]
    #speed_init = data['VEH_SPEED'][1:43189:50]
    # power = []
    data = loadmat('/rwthfs/rz/cluster/home/vv465559/cui/testdata/driving_cycle/wltp2.mat')
    power_init = data['P_DCL_Fahrzeug']['P_custom_wltp_1s']
    speed_init = data['P_DCL_Fahrzeug']['v_custom_wltp_1s']
    power = []
    for i in range(len(power_init)):
        #power.append([i+1, -current[i]*voltage[i]])
        power.append([i, power_init[i][0]])

    #######data_pd = pd.DataFrame(power).to_numpy()
    data_pd = power
    final_data = []
    speed = []
    cost_final= []
    #reward_total = 0
    reward_final = [] #each indicate total reward in one epoch
    for i in range(4):
        for n in range(len(data_pd)):
            final_data.append([n+i*len(data_pd), data_pd[n][1]])  # n start with 0, 17 same trips
            #speed.append(speed_init[n])
            speed.append(speed_init[n][0])
    Env_battery_update = Env_battery([[0, 0]], [[0, 0]], [[0, 0]],[],[],[[0, 0]])
    RL = DeepQNetwork(Env_battery_update.n_actions, Env_battery_update.n_states, learning_rate=0.00025, reward_decay=0.99, e_greedy=1, replace_target_iter=200, memory_size=1000)
    for episode in range(100):
        HP_power_vector = [[0, 0]]
        HE_power_vector = [[0, final_data[0][1]]]
        HEcur = []
        HPcur = []
        cost_total = 0
        reward_total = 0
        #observation = Env_battery([[0, 0]]).reset()
        for time in range(len(final_data)):
            if time == 0:
                observation = Env_battery([[0, 0]], [[0, 0]], [[0, 0]],[],[],[[0, 0]]).reset()
                HE_soc = []
                HP_soc = []
            else:
                print(time)
                Env_battery_update = Env_battery(final_data[0:time], HE_power_vector , HP_power_vector, HEcur, HPcur, speed[0:time])
                #RL = DeepQNetwork(Env_battery_update.n_actions, Env_battery_update.n_states, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, replace_target_iter=200, memory_size=2000)
                action = RL.choose_action(observation)

                # RL take action and get next observation and reward
                observation_, reward, HE_power_vector, HP_power_vector, HEcur, HPcur = Env_battery_update.step(action, time-1)  #####time stamp of action
                print("reward", reward)
                reward_total = reward_total + reward
                
                RL.store_transition(observation, action, reward, observation_)
                
                cost = RL.learn()
                cost_total = cost_total + cost
                # swap observation
                observation = observation_
                HE_soc.append(observation[0]) 
                HP_soc.append(observation[1])
                # break while loop when end of this episode
    # Env_battery.mainloop()
        #RL.plot_cost() 
        cost_final.append(cost_total)
        reward_final.append(reward_total)
		
    plt.plot(np.arange(len(cost_final)),cost_final)
    plt.ylabel('Cost')
    plt.xlabel('Epoch ')
    plt.show()
    plt.savefig('/rwthfs/rz/cluster/home/vv465559/cui/cost_total.png')
    plt.clf()
    
    plt.plot(np.arange(len(reward_final)),reward_final)
    plt.ylabel('Reward')
    plt.xlabel('Epoch ')
    plt.show()
    plt.savefig('/rwthfs/rz/cluster/home/vv465559/cui/reward_total.png')
    plt.clf()
    
    plt.plot(np.arange(len(HE_soc)),HE_soc)
    plt.plot(np.arange(len(HE_soc)),HP_soc)
    plt.ylabel('SOC')
    plt.xlabel('time')
    plt.legend(['HE_SOC', 'HP_SOC'], loc='upper left')
    plt.show()
    plt.savefig('/rwthfs/rz/cluster/home/vv465559/cui/SOC.png')
    plt.clf()
    RL.save()


