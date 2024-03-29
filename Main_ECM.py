from ECM import Battery_ECM as ECM_HE
from ECM_HP import Battery_ECM as ECM_HP
import random as rd
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd


class Env_battery:

    def __init__(self, Power_total, HE_power_vector, HP_power_vector, HEcur, HPcur, Speed_vector ):  ####### self.Current_total is log data or reults from EV model
        self.action_space = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']   #delete the comma in the end 
        self.n_actions = len(self.action_space)
        self.n_states = 4
        #self.Current_total = current_total
        #self.HE_current = 0
        #self.HP_current = self.Current_total[0][1] - self.HE_current
        #self.HE_current_vector = HE_current_vector
        #self.HP_current_vector = HP_current_vector
        self.Power_total = Power_total
        self.HE_power_vector = HE_power_vector
        self.HP_power_vector = HP_power_vector
        self.HE_battery = ECM_HE([0,0])            #using seperate HE and HP battery model
        #print("initialHE", self.HE_current_vector)
        self.HP_battery = ECM_HP([0,0])
        #print("initialHP", self.HP_current_vector)
        self.HE_cell_series =72 #92      #52.2KWH HE and 3.5KWH HP
        self.HE_cell_parallel = 40 #14
        self.HE_cell_num = self.HE_cell_series * self.HE_cell_parallel 
        self.HP_cell_series = 146 #122
        self.HP_cell_parallel = 3 #17
        self.HP_cell_num = self.HP_cell_series * self.HP_cell_parallel 
        self.Speed_vector = Speed_vector
        #self.HE_ratio = self.HE_cell_num / (self.HE_cell_num + self.HP_cell_num )
        #self.HP_ratio = self.HP_cell_num / (self.HE_cell_num + self.HP_cell_num )
        self.HE_P_max_dis = 200000                 #power limit
        self.HE_P_max_cha = -100000
        self.HP_P_max_dis = 200000
        self.HP_P_max_cha = -200000
        self.HE_current = HEcur
        self.HP_current = HPcur

		
    def reset(self):
        self.HE_power_vector = []
        self.HP_power_vector = []
        self.HE_battery = ECM_HE(self.HE_power_vector)
        self.HP_battery = ECM_HP(self.HP_power_vector)
        return (np.array([self.HE_battery.SOC[0][1], self.HP_battery.SOC[0][1],0,0]))  # return the reset initial state


    def step(self, action, num):
        #power distribution
        #print('num',num)
        self.HE_power = self.Power_total[num][1] * action * 0.1
        self.HP_power = self.Power_total[num][1] - self.HE_power
        # former currrent distribution
        #self.HE_current = self.Current_total[num][1] * action * 0.1
        #self.HP_current = self.Current_total[num][1] - self.HE_current
        #HE_cur = self.HE_current / self.HE_cell_parallel
        #HP_cur = self.HP_current / self.HP_cell_parallel
        HP_pow = self.HP_power / self.HP_cell_num
        HE_pow = self.HE_power / self.HE_cell_num
        self.HE_power_vector.append([num, HE_pow])
        self.HP_power_vector.append([num, HP_pow])
        #HP_cur =((ECM_HP.OCV[-1][1]+ECM_HP.V1[-1][1]+ECM_HP.V2[-1][1]) + math.sqrt((ECM_HP.OCV[-1][1]+ECM_HP.V1[-1][1]+ECM_HP.V2[-1][1])**2 + 4*ECM_HP.R0*HP_pow))/(2*ECM_HP.R0)
        #HE_cur =-((ECM_HE.OCV[-1][1]+ECM_HE.V1[-1][1]+ECM_HE.V2[-1][1]) + math.sqrt((ECM_HE.OCV[-1][1]+ECM_HE.V1[-1][1]+ECM_HE.V2[-1][1])**2 + 4*ECM_HE.R0*HE_pow))/(2*ECM_HE.R0)
        #self.HE_current_vector.append([num , HE_cur])
        #self.HP_current_vector.append([num , HP_cur])
        self.HE_battery = ECM_HE(self.HE_power_vector)
        self.HP_battery = ECM_HP(self.HP_power_vector)
        HE_cur = self.HE_battery.cur_dert(num)
        HP_cur = self.HP_battery.cur_dert(num)
        self.HE_current.append([num, HE_cur])
        self.HP_current.append([num, HP_cur])
        #print(HE_current)
        #print(self.HE_current,'self')
        for n in range(len(self.Power_total)):
            #print(len(self.Power_total),'length')
            #HE_current = self.HE_battery.cur_dert(n)
            #HP_current = self.HP_battery.cur_dert(n)
            HE_terminal_volt, HE_soc, HE_v1, HE_v2 = self.HE_battery.twoRCECM(n,self.HE_current)

            HP_terminal_volt, HP_soc, HP_v1, HP_v2 = self.HP_battery.twoRCECM(n,self.HP_current)
        Speed_norm = self.Speed_vector[num]/300     #nomalize of speed and power 
        Power_norm =self.Power_total[num][1]/45000
        next_state = np.array([HE_soc[-1][1], HP_soc[-1][1], Speed_norm, Power_norm])
        #next_state = np.array([HE_soc[-1][1], HP_soc[-1][1]])
        print('#######action',action)
        print(next_state)
	
		############################reward function###################################
        joule_res_HE = (self.HE_current[-1][1]**2 *self.HE_battery.R0 + HE_v1**2/self.HE_battery.R1 + HE_v2**2/self.HE_battery.R2)*self.HE_cell_num  
        joule_res_HP =(self.HP_current[-1][1]**2 *self.HP_battery.R0+ HP_v1**2/self.HP_battery.R1 + HP_v2**2/self.HP_battery.R2)*self.HP_cell_num
        alpha = action/10
        beta = 1 - alpha
        if self.HP_power > self.HP_P_max_dis or self.HP_power < self.HP_P_max_cha : 
            print('reward1') 
            reward_function = -abs(self.HP_power)/1000 #- joule_res
        elif self.HE_power > self.HE_P_max_dis or self.HE_power < self.HE_P_max_cha: 
            print('reward2') 
            reward_function = -abs(self.HE_power)/1000 #- joule_res
        else: 
             # HE battery soc = 0 or HP battery soc = 0
            if HE_cur < -9.6 or  HE_cur > 24 or HP_cur < -200 or  HP_cur > 200 or HE_terminal_volt[-1][1]< 2.5 or HE_terminal_volt[-1][1] > 4.2 or HP_terminal_volt[-1][1]<1.5 or HP_terminal_volt[-1][1]>2.9:
                print('reward3') 
                reward_function = -4000 - (joule_res_HE+joule_res_HP)#- joule_res
            else:
                #if (next_state[0] <= 0.1 or next_state[0] >= 0.9) or (next_state[1] <= 0.2 or next_state[1] >= 0.8 ) :
                    #print('reward4') 
                    #reward_function = -2000
                if next_state[0] <= 0.1:   
                    print('reward4') 
                    reward_function = -(0.1 - next_state[0])*100000 #- joule_res 
                elif next_state[0] >= 0.9 : 
                    print('reward5') 
                    reward_function = -(next_state[0]-0.9)*100000 #- joule_res 
                elif next_state[1] <= 0.2 : 
                    print('reward6') 
                    reward_function = -(0.2 - next_state[1])*100000 #- joule_res 
                elif next_state[1] >= 0.8 :
                    print('reward7') 
                    reward_function = -(next_state[1] - 0.8)*100000 #- joule_res 
                else:
                    #if self.Power_total[num][1] < 0:
                        #if action <3:
                            #print('reward8') 
                            #reward_function = action * 500 - joule_res
                        #else:
                            #print('reward9') 
                            #reward_function = -action * 200 - joule_res
                    #else:
                        #print('reward10') 
                        #reward_function = - (joule_res) -abs(next_state[1]-0.5)*100
                     
                    #if self.Power_total[num][1] > 0: ##discharging 
                                                 
                    print('reward8') 
                    reward_function = - ( joule_res_HE+joule_res_HP) - ( abs(0.5-next_state[1])*20000)
                    #else:  ###charging
                        #print('reward6')					
                        #reward_function = - (alpha*joule_res_HE + beta*joule_res_HP) + (beta * 100) 
            
        print([HE_cur,HP_cur,joule_res_HE,joule_res_HP,self.HE_power,self.HP_power,HE_terminal_volt[-1][1],HP_terminal_volt[-1][1]])

        return next_state, reward_function, self.HE_power_vector, self.HP_power_vector, self.HE_current, self.HP_current
        #return next_state, reward_function


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
