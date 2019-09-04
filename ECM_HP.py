import numpy as np
from scipy.integrate import odeint, quad
from scipy.misc import derivative


class Battery_ECM:
    def __init__(self, i, temp_cell=[0, 300], soc_ini=1, v10=0, v20=0, tamb=298,
                 time_diff=1):  # V10, V20 initial value of V1, V2
        self.I = []
        self.I = i
        # print(self.I)
        self.T = []
        self.T.append(temp_cell)
        self.Tamb = tamb
        self.time_diff = time_diff
        # self.time = []
        # self.time = range(0, t_end, 1)
        # ###########parameter for 2 RC model #######################
        # initial value
        self.SOC_ini = soc_ini
        self.V10 = v10
        self.V20 = v20
        # predifined parameter
        self.Cap = 10440  # unit:As

        # ECM elements parameter
        self.R0 = 4.5588e-3
        self.R1 = 2.5203e-2
        self.R2 = 1.5558e-2
        self.C1 = 2.2447e1
        self.C2 = 3.0020e3
        self.V1 = []
        self.V1.append([0, self.V10])
        self.V2 = []
        self.V2.append([0, self.V20])
        self.OCV = []
        self.OCV.append([0, 2.8])
        self.etha = 0.99
        # self.OCV.append([0, 4.5])            # initial value for OCV
        self.SOC = []
        self.SOC.append([0, self.SOC_ini])
        self.Vt = []
        # ECM discrete state vector space parameter
        self.Ad = []
        self.Ad = np.array([[1, 0, 0], [0, np.exp(-self.time_diff / self.R1 / self.C1), 0],
                            [0, 0, np.exp(-self.time_diff / self.R2 / self.C2)]])

        self.Bd = []
        self.Bd = np.array(
            [[(self.etha * self.time_diff) / self.Cap], [(1 - np.exp(-self.time_diff / self.R1 / self.C1)) * self.R1],
             [(1 - np.exp(-self.time_diff / self.R2 / self.C2)) * self.R2]])

        self.X0 = [[self.SOC_ini], [self.V10], [self.V20]]
        # self.Xk_value = [[self.SOC], [self.V1], [self.V2]]
        self.Xk = []
        self.Xk.append([0, self.X0])
        # self.Xk = np.array(self.Xk)
        # OCV-SOC nonlinearity parameter
        self.K0 = 2.644
        self.K1 = 0.02673
        self.K2 = -0.3406e-3
        self.K3 = 1.848e-6
        self.K4 = 3.706e-2
        self.K5 = -0.2242

        ###############parameter for thermal model###########
        self.V_avg = []
        self.h = 2.4
        self.C_thermal_cell = 0.61

    #################function of ECM############################
    # update main elements and parameter in ECM
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
        OCV_cur = self.K0 + self.K1 * self.SOC[num][1] + self.K2 * np.power(self.SOC[num][1], 2) + self.K3 * np.power(
            self.SOC[num][1], 3) + self.K4 / self.SOC[num][1] + self.K5 * np.log(self.SOC[num][1])
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

    def dV1_cal(self, V1, a):
        dV1dt = (-1 / (self.R1 * self.C1)) * V1 + 1 / self.C1 * a
        return dV1dt

    def dV2_cal(self, V2, a):
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
        self.Xk.append([num + 1, np.dot(self.Ad, self.Xk[num][1]) + np.dot(self.Bd, self.I[num][1])])

        # print(self.Xk)
        self.SOC.append([num + 1, self.Xk[num + 1][1][0][0]])
        self.V1.append([num + 1, self.Xk[num + 1][1][1][0]])
        self.V2.append([num + 1, self.Xk[num + 1][1][2][0]])
        self.Vt.append(
            [num, self.OCV[num + 1][1] + self.V1[num + 1][1] + self.V2[num + 1][1] + self.R0 * self.I[num][1]])
        return self.Vt, self.SOC, self.V1[-1][1], self.V2[-1][1]
