import numpy as np
from scipy.integrate import odeint, quad
from scipy.misc import derivative

class Battery_ECM:
    def __init__(self, i, temp_cell = [0,300], soc_ini = 1, v10 = 0, v20 = 0, tamb = 298, time_diff = 1) : #V10, V20 initial value of V1, V2
        self.I = []
        self.I = i
        #print(self.I)
        self.T = []
        self.T.append(temp_cell)
        self.Tamb = tamb
        self.time_diff = time_diff
        # self.time = []
        # self.time = range(0, t_end, 1)
        # ###########parameter for 2 RC model #######################
        #initial value
        self.SOC_ini = soc_ini
        self.V10 = v10
        self.V20 = v20
        #predifined parameter
        self.Cap = 11448                                              # unit:As

        #ECM elements parameter
        self.R0 = 8.2733e-2
        self.R1 = 1.5115e-2
        self.R2 = 3.6291e-2
        self.C1 = 8.3716e2
        self.C2 = 4.7234e3
        self.V1 = []
        self.V1.append([0, self.V10])
        self.V2 = []
        self.V2.append([0, self.V20])
        self.OCV = []
        self.OCV.append([0, 4.2])
        self.etha = 0.99
        # self.OCV.append([0, 4.5])            # initial value for OCV
        self.SOC = []
        self.SOC.append([0,self.SOC_ini])
        self.Vt = []
        #ECM discrete state vector space parameter
        self.Ad = []
        self.Ad = np.array([[1,0,0],[0, np.exp(-self.time_diff/self.R1/self.C1), 0], [0, 0, np.exp(-self.time_diff/self.R2/self.C2)]])

        self.Bd = []
        self.Bd = np.array([[(self.etha*self.time_diff)/self.Cap], [(1-np.exp(-self.time_diff/self.R1/self.C1))*self.R1], [(1-np.exp(-self.time_diff/self.R2/self.C2))*self.R2]])
        
        self.X0 = [[self.SOC_ini], [self.V10], [self.V20]]
        # self.Xk_value = [[self.SOC], [self.V1], [self.V2]]
        self.Xk = []
        self.Xk.append([0, self.X0])
        # self.Xk = np.array(self.Xk)
        #OCV-SOC nonlinearity parameter
        self.K0 = 3.641
        self.K1 = 0.3696
        self.K2 = -0.6644
        self.K3 = 0.6489
        self.K4 = 1.486e-5
        self.K5 = 0.06265

        ###############parameter for thermal model###########
        self.V_avg = []
        self.h = 2.4
        self.C_thermal_cell = 0.61

    #################function of ECM############################
    #update main elements and parameter in ECM
    # def SOC_upd(self,num ):       #t control the row number point
    # #if I>0    #discharging
    #     delta_SOC = quad(lambda x: self.I[num][1]*self.Cap, 0, self.time_diff)[0]/(self.Cap*3600)
    #     if num == 0:
    #         self.SOC.append([num+1, self.SOC_ini - delta_SOC])
    #     else:
    #         self.SOC.append([num+1, self.SOC[num][1] - delta_SOC])
    #     # print(self.SOC)
    #     return self.SOC

    # def OCV_upd(self,num):
    #     if num == 0:
    #         self.OCV.append([0, 4.5])
    #     else:
    #         OCV_cur = 4.5 * self.SOC[num][1]
    #         self.OCV.append([num, OCV_cur])
    #     return self.OCV

    def ocv_upd(self, num):
        OCV_cur = self.K0 + self.K1 * self.SOC[num][1] + self.K2 * np.power(self.SOC[num][1], 2) + self.K3 * np.power(self.SOC[num][1], 3) + self.K4 / self.SOC[num][1] + self.K5 * np.log(self.SOC[num][1])
        return OCV_cur

    def r0_udp(self):
        self.R0 = 0.0827
        return self.R0


    def c1_upd(self):
        self.C1 = 837
        return self.C1

    def r1_upd(self):
        self.R1 = 0.015
        return self.R1

    def c2_upd(self):
        self.C2 = 4723
        return self.C2

    def r2_upd(self):
        self.R2 = 0.036
        return self.R2

    def dV1_cal(self, V1,  a):
        dV1dt = (-1 / (self.R1 * self.C1)) * V1 + 1 / self.C1 * a
        return dV1dt

    def dV2_cal(self, V2,  a):
        dV2dt = (-1 / (self.R2 * self.C2)) * V2 + 1 / self.C2 * a
        return dV2dt

    def twoRCECM(self, num):
        # self.SOC = self.SOC_upd(num)
        self.OCV.append([num, self.ocv_upd(num)])
        self.R0 = self.r0_udp()
        self.R1 = self.r1_upd()
        self.R2 = self.r2_upd()
        self.C1 = self.c1_upd()
        self.C2 = self.c2_upd()
        # a = self.I[num][1]*self.Cap
        # time_integral = range(0, self.time_diff+1, 1)
        # if num == 0:
        #     self.V1.append([num, quad(self.dV1_cal, self.V10, self.time_diff, args = (a,))[0]])
        #     self.V2.append([num, quad(self.dV2_cal, self.V20, self.time_diff, args = (a,))[0]])
        # else:
        #     self.V1.append([num, quad(self.dV1_cal, self.V1[num-1][1], self.time_diff, args = (a,))[0]])
        #     self.V2.append([num, quad(self.dV2_cal, self.V2[num-1][1], self.time_diff, args = (a,))[0]])
        # self.Vt.append([num, self.OCV[num][1] - self.V1[num][1] - self.V2[num][1] - self.R0*self.I[num][1]])
        self.Xk.append([num+1, np.dot(self.Ad , self.Xk[num][1]) + np.dot(self.Bd , self.I[num][1])])

        # print(self.Xk)
        self.SOC.append([num + 1, self.Xk[num+1][1][0][0]])
        self.V1.append([num + 1, self.Xk[num+1][1][1][0]])
        self.V2.append([num + 1, self.Xk[num+1 ][1][2][0]])
        self.Vt.append([num, self.OCV[num + 1 ][1] + self.V1[num+1][1] + self.V2[num+1][1] + self.R0*self.I[num][1]])
        return self.Vt, self.SOC, self.V1[-1][1], self.V2[-1][1]

    # def V_arv_vs_temp(self, T):  # function of V_arv with regard to temperature
    #     x = np.array(self.OCV)
    #     V_arv = np.average(x[:,1])
    #     V_arv = V_arv * T/ 298
    #     return V_arv
    #
    # def Temp_vs_Time(self, T,   a, num):
    #     x = np.array(self.OCV)
    #     self.C_thermal_cell = self.Tamb + 5               #constant
    #     dQdt = a * (self.OCV[num][1] - np.average(x[:, 1])) + a * T * derivative(self.V_arv_vs_temp, self.T[num][1], dx=self.time_diff)         #average of terminal voltage or OCV???
    #     # print(dQdt)
    #     dTdt = (-self.h*(T - self.Tamb) + dQdt) / self.C_thermal_cell
    #     return dTdt
    #
    # def Thermal_model_simpified(self, num ):
    #     # h is heat convection coefficient, C_thermal is thermal capacity, Tamb is surrounding temperature
    #     # V_arv is function of Temperature
    #     a = self.I[num][1]*self.Cap
    #     # print("{0:.4f}".format(m))
    #     m = quad(self.Temp_vs_Time, 0, self.time_diff, args=(a, num))[0]
    #     print(m)
    #     self.T.append([num, self.T[num][1] + quad(self.Temp_vs_Time, 0, self.time_diff, args = (a, num))[0]])
    #     return self.T


##################################################AgingModel#######################################################

# class
# Aging_model:
#
#
# def __init__(self, ):
#
#
# def HEaging(V, T, I, SOC_ini, SOC_end, V_ini, V_end)
#     # def HEaging(self, SOC_end, V_end)
#
#     # calendrc aging is related to voltage(volts) and absolute temperature(kelvin)
#     alpha_cap = (7.543 * V - 23.75) * np.power(10, 6) * np.exp(-6976 / self.T)
#     alpha_res = (5.270 * V - 16.32) * np.power(10, 5) * np.exp(-5986 / self.T)
#     # cyclinc aging
#     Q = quad(lambda I: I, 0, t1)  # t1 is cycle time
#     deltaDOD = np.abs(SOC_ini - SOC_end)
#     V_arv = 0.5 * (V_ini - V_end)
#     beta_cap = 7.348 * np.power(10, -3) * ((V_arv - 3.667) ** 2) + 7.600 * np.power(10, -4) + 4.081 * np.power(10,
#                                                                                                                -3) * deltaDOD
#     beta_res = 2.153 * np.power(10, -4) * ((V_arv - 3.725) ** 2) - 1.521 * np.power(10, -5) + 2.798 * np.power(10,
#                                                                                                                -4) * deltaDOD
#     # total influence, t in days, Q is ampere hour
#     Cap = 1 - alpha_cap * np.power(t, 0.75) - beta_cap * np.sqrt(Q)
#     Res = 1 + alpha_res * np.power(t, 0.75) - beta_res * np.sqrt(Q)
#     return Cap, Res
#
#
# def HPaging
#
#     return
#
#
# ###############################################update ECM parameter ##############################################
# def ECM_Prara_upd(C.
#
#
# R, Cap, Res )
#
# C_upd = C * Cap
# R_upd = R * Res
# return C_upd, R_upd



