from ECM import Battery_ECM as ECM_HE
from ECM_HP import Battery_ECM as ECM_HP
import random as rd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


class Env_battery:

    def __init__(self, current_total, HE_current_vector, HP_current_vector ):  ####### self.Current_total is log data or reults from EV model
        self.action_space = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']   #delete the comma in the end 
        self.n_actions = len(self.action_space)
        self.n_states = 2
        self.Current_total = current_total
        self.HE_current = 0
        self.HP_current = self.Current_total[0][1] - self.HE_current
        self.HE_current_vector = HE_current_vector
        self.HP_current_vector = HP_current_vector
        self.HE_battery = ECM_HE(self.HE_current_vector)            #using seperate HE and HP battery model
        #print("initialHE", self.HE_current_vector)
        self.HP_battery = ECM_HP(self.HP_current_vector)
        #print("initialHP", self.HP_current_vector)
        self.HE_cell_series =119 #92
        self.HE_cell_parallel = 26 #14
        self.HE_cell_num = self.HE_cell_series * self.HE_cell_parallel 
        self.HP_cell_series = 150 #122
        self.HP_cell_parallel = 1 #17
        self.HP_cell_num = self.HP_cell_series * self.HP_cell_parallel 

    def reset(self):
        self.HE_current_vector = []
        self.HP_currnet_vector = []
        self.HE_battery = ECM_HE(self.HE_current_vector)
        self.HP_battery = ECM_HP(self.HP_current_vector)
        return (np.array([self.HE_battery.SOC[0][1], self.HP_battery.SOC[0][1]]))  # return the reset initial state


    def step(self, action, num):
        self.HE_current = self.Current_total[num][1] * action * 0.1
        #for n in range(self.Current_total):
        #print(action,"###########")
        #print(self.HE_current)
        #print("#####",self.Current_total[num][1])
        self.HP_current = self.Current_total[num][1] - self.HE_current
        HE_cur = self.HE_current / self.HE_cell_parallel
        HP_cur = self.HP_current / self.HP_cell_parallel
        self.HE_current_vector.append([num , HE_cur])
        self.HP_current_vector.append([num , HP_cur])
        self.HE_battery = ECM_HE(self.HE_current_vector)
        self.HP_battery = ECM_HP(self.HP_current_vector)
        #print(self.HE_current_vector)
        #print(self.HP_current_vector)
        for n in range(len(self.HE_current_vector)):
            HE_terminal_volt, HE_soc, HE_v1, HE_v2 = self.HE_battery.twoRCECM(n )
            HP_terminal_volt, HP_soc, HP_v1, HP_v2 = self.HP_battery.twoRCECM(n )
        next_state = np.array([HE_soc[-1][1], HP_soc[-1][1]])
        #term_voltage = np.array([HE_terminal_volt[-1][1], HP_terminal_volt[-1][1]])
        #print(action)
        print(next_state)
        #print(self.HE_current_vector)
        #print(self.HP_current_vector)
        # reward function
		
        if next_state[0] <= 0.1 or next_state[1] <= 0.1:  # HE battery soc = 0 or HP battery soc = 0
            reward_function = -1000
        # elif next_state[1] <= 0.1:  # 
            # reward_function = - 100
        else:
            #reward_function = -(HE_cur *(self.HE_battery.OCV[-1][1] - HE_terminal_volt[-1][1]) + HP_cur *(self.HP_battery.OCV[-1][1] - HP_terminal_volt[-1][1]))
            reward_function = - (HE_cur**2 *self.HE_battery.R0 + HE_v1**2/self.HE_battery.R1 + HE_v2**2/self.HE_battery.R2  + HP_cur**2 *self.HP_battery.R0+ HP_v1**2/self.HP_battery.R1 + HP_v2**2/self.HP_battery.R2)
        # if len(self.HE_current_vector) == len(self.Current_total):
        #     done = True
        # elif len(self.HE_current_vector) > len(self.Current_total):
        #     print('there is error with the current vector iteration')
        #     done = False
        # else:
        #     done = False

        return next_state, reward_function, self.HE_current_vector, self.HP_current_vector



# ######################plot corresponding pictures####################################
# plt.figure()
# plt.subplot(3, 1, 1)
# plt.plot(input[:, 1])
# plt.grid(True)
# plt.ylabel('I');
# plt.xlabel('time')
# plt.subplot(3, 1, 2)
# plt.grid(True)
# plt.plot(terminal_voltage[:, 1])
# plt.ylabel('voltage');
# plt.xlabel('time')
# # plt.subplot(3,1,3)
# # plt.grid(True)
# # plt.plot(battery_temperature[:,1])
# # plt.ylabel('battery_temperature'); plt.xlabel('time')
# # plt.savefig('/home/cuihan/Downloads/masterthesis/TestResults/ECM_test.png')
# plt.show()
