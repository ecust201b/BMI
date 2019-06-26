from model import BMI_Model
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import scipy.io as sio
import math
from keras.models import load_model
from sklearn.externals import joblib


def computeMetrics(dataDic):
    cnt = 0
    while cnt < len(dataDic['P1']) and dataDic['P1'][cnt] < dataDic['TI'][cnt]:
        cnt += 1
    if cnt != len(dataDic['P1']):
        time = cnt
        lll = (max(dataDic['P1']) - dataDic['TI'][cnt]) / dataDic['TI'][cnt]
        overshoot = round(lll, 4)
    else:
        time = 0
        overshoot = 0
    E = 0
    for a in dataDic['DP1']:
        E += a ** 2
    E = round(E, 4)
    energy = E
    return overshoot, time, energy


def computeMetrics_Low(dataDic):
    cnt = 0
    while cnt < len(dataDic['P1']) and dataDic['P1'][cnt] > dataDic['TI'][cnt]:
        cnt += 1
    if cnt != len(dataDic['P1']):
        time = cnt
        lll = (max(dataDic['P1']) - dataDic['TI'][cnt]) / dataDic['TI'][cnt]
        overshoot = round(lll, 4)
    else:
        time = 0
        overshoot = 0
    E = 0
    for a in dataDic['DP1']:
        E += a ** 2
    E = round(E, 4)
    energy = E
    return overshoot, time, energy


def modelMetric(constant, initalVal, runType, factorList, *factorname):
    overshoot = []
    time = []
    energy = []
    for coff in factorList:
        if isinstance(coff, type(())):
            for key, value in zip(factorname, coff):
                initalVal[key] = value
        initalVal[factorname[0]] = coff
        print(initalVal['Z'])
        modelDynamicg = BMI_Model(constant, initalVal)
        if runType == 'dynamic':
            dataDynamicg = modelDynamicg.modelRunDynamic()
        elif runType == 'static':
            dataDynamicg = modelDynamicg.modelRun()
        if dataDynamicg['TI'][0] >= dataDynamicg['P1'][0]:
            overshoot1, time1, energy1 = computeMetrics(dataDynamicg)
        else:
            overshoot1, time1, energy1 = computeMetrics_Low(dataDynamicg)
        print(computeMetrics(dataDynamicg))
        overshoot.append(overshoot1)
        time.append(time1)
        energy.append(energy1)
    return overshoot, time, energy


def mse(target, output):
    sqsum = 0
    for x, y in zip(target, output):
        sqsum += (x - y) ** 2
    return sqsum / (len(target))


def COR(target, output):
    sigmaXY = 0
    sigmaX = 0
    sigmaY = 0
    M_t = sum(target) / len(target)
    M_o = sum(output) / len(output)
    for x, y in zip(target, output):
        sigmaXY += (x - M_t) * (y - M_o)
        sigmaX += (x - M_t) ** 2
        sigmaY += (y - M_o) ** 2
    return sigmaXY / math.sqrt(sigmaX * sigmaY)


def rSquared(target, output):
    res = 0
    tot = 0
    M_t = sum(target) / len(target)
    for x, y in zip(target, output):
        res += (x - y) ** 2
        tot += (x - M_t) ** 2
    return 1 - res / tot


start = datetime.now()
constant = {}
constant['Ix'] = 200
constant['Vx'] = 10
constant['Br'] = 0.1
constant['Bu'] = 0.01
# constant['row'] = 0.04
# constant['Thetax'] = 0.5
constant['row'] = 0.0
constant['Thetax'] = 0.0
constant['etax'] = 0.7
constant['lamda1x'] = 150
constant['lamda2x'] = 10
constant['Gammax'] = 0.001
constant['Cx'] = 25
constant['epsx'] = 0.05
constant['xix'] = 4
constant['hx'] = 0.01
constant['TI'] = 0.7
constant['Eix'] = 0
constant['delta'] = 0.1
constant['v'] = 0.15
initalVal = {}
initalVal['x1'] = 0.5
initalVal['x2'] = 0.5
initalVal['y1'] = 0.5
initalVal['y2'] = 0.5
initalVal['g1'] = 0
initalVal['p1'] = 0.5
initalVal['dp1'] = 0
initalVal['p2'] = 0.5
initalVal['dp2'] = 0
initalVal['q1'] = 0
initalVal['q2'] = 0
initalVal['f1'] = 0
initalVal['f2'] = 0
initalVal['s1_1'] = 0
initalVal['s1_2'] = 0
initalVal['s2_1'] = 0
initalVal['s2_2'] = 0
initalVal['c1'] = 0
initalVal['c2'] = 0
initalVal['y0'] = 0
initalVal['vObj'] = -0.001
initalVal['Z'] = 3
initalVal['g0'] = 0.75
stdc = joblib.load('stdsc_python_mlp.m')
model1 = load_model('mlp_x1.h5')
stdc2 = joblib.load('stdsc_python_mlp_x15f.m')
model2 = load_model('mlp_x1_5f.h5')
model_10f = load_model('mlp_x1_10f.h5')
model3 = load_model('model_mlp_5_to_x1.h5')
stdc3 = joblib.load('stdsc_mlp_5_to_x1.m')
model4 = load_model('model_mlp_5_to_x1_plus_x1.h5')
stdc4 = joblib.load('stdsc_mlp_5_to_x1_plus_x1.m')
model5 = load_model('model_mlp_5_to_x1_plus_x1_233600.h5')
stdc5 = joblib.load('stdsc_mlp_5_to_x1_plus_x1_233600.m')

# ----------------Weiner and Kalman Param----------------- #
matfn1 = u'D:\\学习\\OneDrive\\研究生论文\\毕业论文相关\\脑机\\程序\\原模型\\训练权重Wdelta为1.mat'
matfn2 = u'D:\\学习\\OneDrive\\研究生论文\\毕业论文相关\\脑机\\程序\\原模型\\KalmanParam.mat'
dataK = sio.loadmat(matfn2)
dataW = sio.loadmat(matfn1)
W = np.array(dataW['w'])
A = np.array(dataK['A'])
C = np.array(dataK['C'])
Q = np.array(dataK['Q'])
R = np.array(dataK['R'])
# -----------------Model build-------------------- 动态(dynamic decoder + model1) 静态(batch0 decoder + model4 / model_5fx1)#
model_mlp_8f = BMI_Model(constant, initalVal)
data_elm = model_mlp_8f.modelRun_mlp(model4, stdc4)
# model_mlp_5fx1 = BMI_Model(constant, initalVal)
# data_mlp_5fx1 = model_mlp_5fx1.modelRun_mlp_5fx1(model4, stdc4)
model = BMI_Model(constant, initalVal)
timeZone = model.ns
dataAll = model.modelRun()
modelDynamic = BMI_Model(constant, initalVal)
dataDynamic = modelDynamic.modelRunDynamic()
# modelDecoder = BMI_Model(constant, initalVal)
# dataDecoder = modelDecoder.modelRunDecoder()
# dataDecoderD = modelDecoder.modelRunDecoderDynamic()
# modelWeiner = BMI_Model(constant, initalVal)
# dataWeinerN = modelWeiner.modelRun_Weiner(W)
# modelWeinerOriginal = BMI_Model(constant, initalVal)
# dataWeinerO = modelWeinerOriginal.modelRun_WeinerOriginal(W)
#---------test picture------------#
plt.plot(dataDynamic['X1'], linewidth=2, label='original $x_{i}$')
plt.plot(data_elm['X1'], linewidth=2, label='ANN + UKF $x_{i}$')
plt.xlabel('time(s)', fontsize=12)
plt.ylabel('Position', fontsize=12)
plt.legend(loc='best', fontsize=12)
plt.show()

end = datetime.now()
print('time:', (end - start).seconds)
timeline = [x / 100 for x in range(timeZone)]

# # ---------------C4.1----------------- #
# fig10, ax = plt.subplots(ncols=1, nrows=2)

# ax[0].plot(timeline, dataAll['P1'], linewidth=2, label='Original Model')
# ax[0].plot(timeline, dataDecoder['P1'], linewidth=2, label='RNN Decoder Model', linestyle='--', color='red')
# ax[0].plot(timeline, dataWeinerO['P1'], linewidth=2, label='Weiner Decoder Model', linestyle='-.', color='green')
# ax[0].plot(timeline, dataAll['TI'], linewidth=2, label='Target', linestyle=':', color='coral')
# ax[0].set_xlabel('time(s)', fontsize=12)
# ax[0].set_ylabel('$p_{i}$', fontsize=12)
# ax[0].legend(fontsize=12, loc='best')
# ax[0].set_title('$p_{i}$ compare', fontsize=12)

# ax[1].plot(timeline, dataAll['DP1'], linewidth=2, label='Original Model')
# ax[1].plot(timeline, dataDecoder['DP1'], linewidth=2, label='RNN Decoder Model', linestyle='--', color='red')
# ax[1].plot(timeline, dataWeinerO['DP1'], linewidth=2, label='Weiner Decoder Model', linestyle='-.', color='green')
# ax[1].set_xlabel('time(s)', fontsize=12)
# ax[1].set_ylabel('$dp_{i}$', fontsize=12)
# ax[1].legend(fontsize=12, loc='best')
# ax[1].set_title('$dp_{i}$ compare', fontsize=12)


# # ---------------C4.4.1----------------- #

# ---------------static reaching pi picture----------------- #
# fig1 = plt.figure('HandP')
# plt.plot(timeline, dataAll['P1'],
#          linewidth=2, linestyle='-',
#          label='Original Model', color='blue')
# plt.plot(timeline, data_elm['P1'], linewidth=2, label='Decoder Model', linestyle='--', color='red')
# plt.plot(timeline, dataAll['TI'],
#          linewidth=2, linestyle=':',
#          label='target', color='coral')
# plt.xlabel('time(s)')
# plt.ylabel('$p_{i}$', fontsize=12)
# plt.legend(loc='best')
# fig1.show()

# # ---------------static reaching dpi picture----------------- #
# fig3 = plt.figure('$dP_{i}$ compare')
# plt.plot(timeline, dataAll['DP1'], linewidth=2,
#          label='Original Model', color='blue')
# plt.plot(timeline, data_elm['DP1'], linewidth=2, label='Decoder Model', linestyle='--', color='red')
# plt.xlabel('time(s)')
# plt.ylabel('$dP_{i}$')
# plt.legend(loc='best')
# fig3.show()

# print(computeMetrics(dataAll))
# print(computeMetrics(data_elm))

# # ---------------C4.4.2----------------- #

# # ---------------dynamic reaching pi picture----------------- #
fig1 = plt.figure('HandP')
plt.plot(timeline, dataDynamic['P1'],
         linewidth=2, linestyle='-',
         label='Original Model', color='blue')
plt.plot(timeline, data_elm['P1'], linewidth=2, label='Decoder Model', linestyle='--', color='red')
plt.plot(timeline, dataDynamic['TI'],
         linewidth=2, linestyle=':',
         label='target', color='coral')
plt.xlabel('time(s)')
plt.ylabel('$p_{i}$', fontsize=12)
plt.legend(loc='best')
fig1.show()

# --------------- dynamic reaching dpi picture----------------- #
fig3 = plt.figure('$dP_{i}$ compare')
plt.plot(timeline, dataDynamic['DP1'], linewidth=2,
         label='Original Model', color='blue')
plt.plot(timeline, data_elm['DP1'], linewidth=2, label='Decoder Model', linestyle='--', color='red')
plt.xlabel('time(s)')
plt.ylabel('$dP_{i}$')
plt.legend(loc='best')
fig3.show()

print(computeMetrics(dataAll))
print(computeMetrics(data_elm))

# # ---------------C3.4.1----------------- #

# # ---------------static reaching pi picture----------------- #
# fig1 = plt.figure('HandP')
# plt.plot(timeline, dataAll['P1'],
#          linewidth=2, linestyle='-',
#          label='Original Model', color='blue')
# plt.plot(timeline, dataDecoder['P1'], linewidth=2, label='RNN Decoder Model', linestyle='--', color='red')
# plt.plot(timeline, dataWeinerO['P1'], linewidth=2, label='Weiner Decoder Model', linestyle='-.', color='green')
# plt.plot(timeline, dataAll['TI'],
#          linewidth=2, linestyle=':',
#          label='target', color='coral')
# plt.xlabel('time(s)')
# plt.ylabel('$p_{i}$', fontsize=12)
# plt.legend(loc='best')
# fig1.show()

# # ---------------C3 static reaching dpi picture----------------- #
# fig3 = plt.figure('$dP_{i}$ compare')
# plt.plot(timeline, dataAll['DP1'], linewidth=2,
#          label='Original Model', color='blue')
# plt.plot(timeline, dataDecoder['DP1'], linewidth=2, label='RNN Decoder Model', linestyle='--', color='red')
# plt.plot(timeline, dataWeinerO['DP1'], linewidth=2, label='Weiner Decoder Model', linestyle='-.', color='green')
# plt.xlabel('time(s)')
# plt.ylabel('$dP_{i}$')
# plt.legend(loc='best')
# fig3.show()

# # ---------------C3 static reaching y0 picture----------------- #
# fig4 = plt.figure('y0 compare')
# plt.plot(timeline, dataAll['Y0'],linewidth=2,
#          label='Original Model', color='blue')
# plt.plot(timeline, dataDecoder['Y0'],
#          linewidth=2, linestyle='--', color='red',
#          label='RNN Decoder Model')
# plt.plot(timeline, dataWeinerO['Y0'],
#          linewidth=2, linestyle='-.', color='green',
#          label='Weiner Decoder Model')
# plt.xlabel('time(s)')
# plt.ylabel('$\Delta M$')
# plt.legend()
# fig4.show()

#--------static reaching metric-----------#
# print(mse(dataAll['Y0'], dataDecoder['Y0']),
#       COR(dataAll['Y0'], dataDecoder['Y0']),
#       rSquared(dataAll['Y0'], dataDecoder['Y0']))
# print(mse(dataAll['Y0'], dataWeinerO['Y0']),
#       COR(dataAll['Y0'], dataWeinerO['Y0']),
#       rSquared(dataAll['Y0'], dataWeinerO['Y0']))
# print(computeMetrics(dataAll))
# print(computeMetrics(dataDecoder))
# print(computeMetrics(dataWeinerO))

# ---------------C3 static reaching other vector picture----------------- #
# fig10, ax = plt.subplots(figsize=(9,6), ncols=2, nrows=2)

# ax[0][0].plot(timeline, dataAll['R1'], linewidth=2, label='Original Model')
# ax[0][0].plot(timeline, dataDecoder['R1'], linewidth=2, label='RNN Decoder Model', linestyle='--', color='red')
# ax[0][0].plot(timeline, dataWeinerO['R1'], linewidth=2, label='Weiner Decoder Model', linestyle='-.', color='green')
# ax[0][0].set_xlabel('time(s)', fontsize=12)
# ax[0][0].set_ylabel('$r_{i}$', fontsize=12)
# ax[0][0].legend(fontsize=12, loc='best')
# ax[0][0].set_title('$r_{i}$ compare', fontsize=12)

# ax[0][1].plot(timeline, dataAll['U1'], linewidth=2, label='Original Model')
# ax[0][1].plot(timeline, dataDecoder['U1'], linewidth=2, label='RNN Decoder Model', linestyle='--', color='red')
# ax[0][1].plot(timeline, dataWeinerO['U1'], linewidth=2, label='Weiner Decoder Model', linestyle='-.', color='green')
# ax[0][1].set_xlabel('time(s)', fontsize=12)
# ax[0][1].set_ylabel('$u_{i}$', fontsize=12)
# ax[0][1].legend(fontsize=12, loc='best')
# ax[0][1].set_title('$u_{i}$ compare', fontsize=12)

# ax[1][0].plot(timeline, dataAll['Y1'], linewidth=2, label='Original Model')
# ax[1][0].plot(timeline, dataDecoder['Y1'], linewidth=2, label='RNN Decoder Model', linestyle='--', color='red')
# ax[1][0].plot(timeline, dataWeinerO['Y1'], linewidth=2, label='Weiner Decoder Model', linestyle='-.', color='green')
# ax[1][0].set_xlabel('time(s)', fontsize=12)
# ax[1][0].set_ylabel('$y_{i}$', fontsize=12)
# ax[1][0].legend(fontsize=12, loc='best')
# ax[1][0].set_title('$y_{i}$ compare', fontsize=12)

# ax[1][1].plot(timeline, dataAll['X1'], linewidth=2, label='Original Model')
# ax[1][1].plot(timeline, dataDecoder['X1'], linewidth=2, label='RNN Decoder Model', linestyle='--', color='red')
# ax[1][1].plot(timeline, dataWeinerO['X1'], linewidth=2, label='Weiner Decoder Model', linestyle='-.', color='green')
# ax[1][1].set_xlabel('time(s)', fontsize=12)
# ax[1][1].set_ylabel('$x_{i}$', fontsize=12)
# ax[1][1].legend(fontsize=12, loc='best')
# ax[1][1].set_title('$x_{i}$ compare', fontsize=12)

# fig10.show()


# ---------------C3.4.2----------------- #

# # ---------------C3 dynamic reaching pi picture----------------- #
# fig1 = plt.figure('HandP')
# plt.plot(timeline, dataDynamic['P1'],
#          linewidth=2, linestyle='-',
#          label='Original Model', color='blue')
# plt.plot(timeline, dataDecoderD['P1'], linewidth=2, label='RNN Decoder Model', linestyle='--', color='red')
# plt.plot(timeline, dataWeinerN['P1'], linewidth=2, label='Weiner Decoder Model', linestyle='-.', color='green')
# plt.plot(timeline, dataDynamic['TI'],
#          linewidth=2, linestyle=':',
#          label='target', color='coral')
# plt.xlabel('time(s)')
# plt.ylabel('$p_{i}$', fontsize=12)
# plt.legend(loc='best')
# fig1.show()

# # ---------------C3 dynamic reaching dpi picture----------------- #
# fig3 = plt.figure('$dP_{i}$ compare')
# plt.plot(timeline, dataDynamic['DP1'], linewidth=2,
#          label='Original Model', color='blue')
# plt.plot(timeline, dataDecoderD['DP1'], linewidth=2, label='RNN Decoder Model', linestyle='--', color='red')
# plt.plot(timeline, dataWeinerN['DP1'], linewidth=2, label='Weiner Decoder Model', linestyle='-.', color='green')
# plt.xlabel('time(s)')
# plt.ylabel('$dP_{i}$')
# plt.legend(loc='best')
# fig3.show()

# # ---------------C3 dynamic reaching y0 picture----------------- #
# fig4 = plt.figure('y0 compare')
# plt.plot(timeline, dataDynamic['Y0'],linewidth=2,
#          label='Original Model', color='blue')
# plt.plot(timeline, dataDecoderD['Y0'],
#          linewidth=2, linestyle='--', color='red',
#          label='RNN Decoder Model')
# plt.plot(timeline, dataWeinerN['Y0'],
#          linewidth=2, linestyle='-.', color='green',
#          label='Weiner Decoder Model')
# plt.xlabel('time(s)')
# plt.ylabel('$\Delta M$')
# plt.legend()
# fig4.show()

# # ---------------C3 dynamic reaching other vector picture----------------- #
# fig10, ax = plt.subplots(figsize=(9,6), ncols=2, nrows=2)

# ax[0][0].plot(timeline, dataDynamic['R1'], linewidth=2, label='Original Model')
# ax[0][0].plot(timeline, dataDecoderD['R1'], linewidth=2, label='RNN Decoder Model', linestyle='--', color='red')
# ax[0][0].plot(timeline, dataWeinerN['R1'], linewidth=2, label='Weiner Decoder Model', linestyle='-.', color='green')
# ax[0][0].set_xlabel('time(s)', fontsize=12)
# ax[0][0].set_ylabel('$r_{i}$', fontsize=12)
# ax[0][0].legend(fontsize=12, loc='best')
# ax[0][0].set_title('$r_{i}$ compare', fontsize=12)

# ax[0][1].plot(timeline, dataDynamic['U1'], linewidth=2, label='Original Model')
# ax[0][1].plot(timeline, dataDecoderD['U1'], linewidth=2, label='RNN Decoder Model', linestyle='--', color='red')
# ax[0][1].plot(timeline, dataWeinerN['U1'], linewidth=2, label='Weiner Decoder Model', linestyle='-.', color='green')
# ax[0][1].set_xlabel('time(s)', fontsize=12)
# ax[0][1].set_ylabel('$u_{i}$', fontsize=12)
# ax[0][1].legend(fontsize=12, loc='best')
# ax[0][1].set_title('$u_{i}$ compare', fontsize=12)

# ax[1][0].plot(timeline, dataDynamic['Y1'], linewidth=2, label='Original Model')
# ax[1][0].plot(timeline, dataDecoderD['Y1'], linewidth=2, label='RNN Decoder Model', linestyle='--', color='red')
# ax[1][0].plot(timeline, dataWeinerN['Y1'], linewidth=2, label='Weiner Decoder Model', linestyle='-.', color='green')
# ax[1][0].set_xlabel('time(s)', fontsize=12)
# ax[1][0].set_ylabel('$y_{i}$', fontsize=12)
# ax[1][0].legend(fontsize=12, loc='best')
# ax[1][0].set_title('$y_{i}$ compare', fontsize=12)

# ax[1][1].plot(timeline, dataDynamic['X1'], linewidth=2, label='Original Model')
# ax[1][1].plot(timeline, dataDecoderD['X1'], linewidth=2, label='RNN Decoder Model', linestyle='--', color='red')
# ax[1][1].plot(timeline, dataWeinerN['X1'], linewidth=2, label='Weiner Decoder Model', linestyle='-.', color='green')
# ax[1][1].set_xlabel('time(s)', fontsize=12)
# ax[1][1].set_ylabel('$x_{i}$', fontsize=12)
# ax[1][1].legend(fontsize=12, loc='best')
# ax[1][1].set_title('$x_{i}$ compare', fontsize=12)

# fig10.show()


# #--------dynamic reaching metric-----------#
# print(mse(dataDynamic['Y0'], dataDecoderD['Y0']),
#       COR(dataDynamic['Y0'], dataDecoderD['Y0']),
#       rSquared(dataDynamic['Y0'], dataDecoderD['Y0']))
# print(mse(dataDynamic['Y0'], dataWeinerN['Y0']),
#       COR(dataDynamic['Y0'], dataWeinerN['Y0']),
#       rSquared(dataDynamic['Y0'], dataWeinerN['Y0']))
# print(computeMetrics(dataDynamic))
# print(computeMetrics(dataDecoderD))
# print(computeMetrics(dataWeinerN))


#-------------------C2-----------------#


# # ---------------C2 Fi picture----------------- #
# fig6 = plt.figure('$f_{i}$ compare')
# plt.plot(timeline, dataAll['F1'], linewidth=2,
#          label='$f_{i}$ Original Model')
# # plt.plot(timeline, dataDecoder['DP1'], linewidth=3, label='dPi Decoder Model')
# plt.xlabel('time(s)', fontsize=12)
# plt.ylabel('$f_{i}$', fontsize=12)
# # plt.title('dPi comparison in Static Reaching Task')
# plt.legend(loc='best', fontsize=12)
# fig6.show()

# # ---------------C2 Qi picture----------------- #
# fig7 = plt.figure('$q_{i}$ compare')
# plt.plot(timeline, dataAll['Q1'], linewidth=2,
#          label='$q_{i}$ Original Model')
# # plt.plot(timeline, dataDecoder['DP1'], linewidth=3, label='dPi Decoder Model')
# plt.xlabel('time(s)', fontsize=12)
# plt.ylabel('$q_{i}$', fontsize=12)
# # plt.title('dPi comparison in Static Reaching Task')
# plt.legend(loc='best', fontsize=12)
# fig7.show()

# # --------------- C2 pi and xi picture----------------- #
# fig8 = plt.figure('compare')
# plt.plot(timeline, dataAll['P1'], linewidth=2,
#          label='$p_{i}$ Original Model')
# plt.plot(timeline, dataAll['X1'], linewidth=2, linestyle='--',
#          label='$x_{i}$ Original Model')
# # plt.plot(timeline, dataDecoder['DP1'], linewidth=3, label='dPi Decoder Model')
# plt.xlabel('time(s)', fontsize=12)
# plt.ylabel('Position', fontsize=12)
# # plt.title('dPi comparison in Static Reaching Task')
# plt.legend(loc='best', fontsize=12)
# fig8.show()

# --------------- xi picture----------------- #
# fig9 = plt.figure('xi')
# plt.plot(timeline, dataAll['X1'], linewidth=2,
#          label='$p_{i}$ Original Model')
# plt.plot(timeline, data_mlp_5fx1['X1'], linewidth=2, linestyle='--',
#          label='$x_{i}$ ANN 10F Model')
# plt.plot(timeline, data_elm['X1'], linewidth=2, linestyle='--',
#          label='$x_{i}$ ANN 8f Model') 
# # plt.plot(timeline, dataDecoder['DP1'], linewidth=3, label='dPi Decoder Model')
# plt.xlabel('time(s)', fontsize=12)
# plt.ylabel('Position', fontsize=12)
# # plt.title('dPi comparison in Static Reaching Task')
# plt.legend(loc='best', fontsize=12)
# fig8.show()


# ---------------C2 original model result----------------- #

# fig5, ax = plt.subplots(figsize=(9,6), ncols=2, nrows=4)

# ax[0][0].plot(timeline, dataAll['R1'], linewidth=2, label='$r_{i}$ Original Model')
# ax[0][0].set_xlabel('time(s)', fontsize=12)
# ax[0][0].set_ylabel('$r_{i}$', fontsize=12)
# ax[0][0].legend(fontsize=12)

# ax[0][1].plot(timeline, dataAll['U1'], linewidth=2, label='$u_{i}$ Original Model')
# ax[0][1].set_xlabel('time(s)', fontsize=12)
# ax[0][1].set_ylabel('$u_{i}$', fontsize=12)
# ax[0][1].legend(fontsize=12)

# ax[1][0].plot(timeline, dataAll['Y1'], linewidth=2, label='$y_{i}$ Original Model')
# ax[1][0].set_xlabel('time(s)', fontsize=12)
# ax[1][0].set_ylabel('$y_{i}$', fontsize=12)
# ax[1][0].legend(fontsize=12)

# ax[1][1].plot(timeline, dataAll['A1'], linewidth=2, label='$a_{i}$ Original Model')
# ax[1][1].set_xlabel('time(s)', fontsize=12)
# ax[1][1].set_ylabel('$a_{i}$', fontsize=12)
# ax[1][1].legend(fontsize=12)

# ax[2][0].plot(timeline, dataAll['P1'], linewidth=2, label='$p_{i}$ Original Model')
# ax[2][0].set_xlabel('time(s)', fontsize=12)
# ax[2][0].set_ylabel('$p_{i}$', fontsize=12)
# ax[2][0].legend(fontsize=12)

# ax[2][1].plot(timeline, dataAll['X1'], linewidth=2, label='$x_{i}$ Original Model')
# ax[2][1].set_xlabel('time(s)', fontsize=12)
# ax[2][1].set_ylabel('$x_{i}$', fontsize=12)
# ax[2][1].legend(fontsize=12)

# ax[3][0].plot(timeline, dataAll['DP1'], linewidth=2, label='$dp_{i}$ Original Model')
# ax[3][0].set_xlabel('time(s)', fontsize=12)
# ax[3][0].set_ylabel('$dp_{i}$', fontsize=12)
# ax[3][0].legend(fontsize=12)

# ax[3][1].plot(timeline, dataAll['Y0'], linewidth=2, label='$\Delta M$ Original Model')
# ax[3][1].set_xlabel('time(s)', fontsize=12)
# ax[3][1].set_ylabel('$\Delta M$', fontsize=12)
# ax[3][1].legend(fontsize=12)

# fig5.show()

# plt.show()

plt.show()

# ----------table data------------------#
# initalZ = [0, 0.5, 1, 3, 3.5, 5, 7]
# initalg = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 1, 2, 3, 4]
# pairs = [(0.7, -0.003), (0.7, -0.005), (0.7, 0.001)]
# pairs2 = [(0.3, 0.001), (0.3, -0.001)]
# overshoot = []
# time = []
# energy = []

# -----------different g0, zeta metrics-------------#
# overshoot, time, energy = modelMetric(constant, initalVal, 'dynamic', initalZ, 'Z')
# overshootS, timeS, energyS = modelMetric(constant, initalVal, 'static', initalZ, 'Z')
# # overshoot, time, energy = modelMetric(constant, initalVal, 'dynamic', pairs2, 'TI', 'vObj')
# # overshootS, timeS, energyS = modelMetric(constant, initalVal, 'static', pairs2, 'TI', 'vObj')
# print('-------------improved model-------------')
# print('catching time:', time)
# print('overshoot:', overshoot)
# print('energy:', energy)
# print('-------------original model-------------')
# print('catching time:', timeS)
# print('overshoot:', overshootS)
# print('energy:', energyS)


# -----------different Zeta-------------#
# for coff in initalZ:
#     initalVal['Z'] = coff
#     modelDynamicZ = BMI_Model(constant, initalVal)
#     dataDynamicZ = modelDynamicZ.modelRunDynamic()
#     cnt = 0
#     while dataDynamicZ['P1'][cnt] < dataDynamicZ['TI'][cnt]:
#         cnt += 1
#     if cnt != len(dataDynamicZ['P1']) - 1:
#         time.append(cnt)
#         overshoot.append((max(dataDynamicZ['P1']) -  dataDynamicZ['TI'][cnt]) / dataDynamicZ['TI'][cnt])
#     else:
#         time.append(0)
#         overshoot.append(0)
# print(time)
# print(overshoot)


# ---------------pi picture template----------------- #
# fig1 = plt.figure('HandP')
# plt.plot(timeline, dataDynamic['P1'],
#          linewidth=2, color='blue', linestyle='-',
#          label='Pi Improved Model')
# plt.plot(timeline, dataAll['P1'],
#          linewidth=2, linestyle='-',
#          label='Original Model', color='blue')
# plt.plot(timeline, dataDecoder['P1'], linewidth=2, label='RNN Decoder Model', linestyle='--', color='red')
# plt.plot(timeline, data_elm['P1'], linewidth=2, label='mlp_8f Decoder Model')
# plt.plot(timeline, data_mlp_5fx1['P1'], linewidth=2, label='mlp_10fx1 Decoder Model')
# plt.plot(timeline, dataWeinerO['P1'], linewidth=2, label='Weiner Decoder Model', linestyle='-.', color='green')

# plt.plot(timeline, dataAll['TI'],
#          linewidth=2, linestyle=':',
#          label='target', color='coral')
# plt.xlabel('time(s)')
# plt.ylabel('$p_{i}$', fontsize=12)
# plt.title('Limb Trajectories in Static Reaching Task')
# plt.legend(loc='best')
# plt.axes([0.4, 0.35, 0.45, 0.3])
# plt.plot(timeline[90: 150], dataDynamic['TI'][90: 150],
#          color='green', linestyle=':', linewidth=2,
#          label='Target')
# plt.plot(timeline[90: 150], dataAll['P1'][90: 150],
#          color='red', linestyle='--', linewidth=2,
#          label='Pi Original Model')
# plt.plot(timeline[90: 150], dataDynamic['P1'][90: 150],
#          color='blue', linestyle='-', linewidth=2,
#          label='Pi Improved Model')
# plt.title('0.9s-1.5s Zoom In')
# fig1.show()

# ---------------dpi picture template----------------- #
# fig3 = plt.figure('$dP_{i}$ compare')
# plt.plot(timeline, dataDynamic['DP1'], linewidth=2,
#          color='blue', linestyle='-',
#          label='dPi Improved Model')
# plt.plot(timeline, dataAll['DP1'], linewidth=2,linestyle='--',
#          label='$dP_{i}$ Original Model')
# plt.plot(timeline, dataAll['DP1'], linewidth=2,
#          label='Original Model', color='blue')
# plt.plot(timeline, dataDecoder['DP1'], linewidth=2, label='RNN Decoder Model', linestyle='--', color='red')
# plt.plot(timeline, dataWeinerO['DP1'], linewidth=2, label='Weiner Decoder Model', linestyle='-.', color='green')
# plt.xlabel('time(s)')
# plt.ylabel('$dP_{i}$')
# plt.title('dPi comparison in Static Reaching Task')
# plt.legend(loc='best')
# fig3.show()
# plt.axes([0.4, 0.4, 0.45, 0.3])
# fmt = '%.1f*10^(-4)'
# # yticks = mtick.FormatStrFormatter(fmt)
# plt.plot(timeline[80: 130], dataAll['DP1'][80: 130],
#          color='red', linestyle='--', linewidth=2,
#          label='Pi Original Model')
# plt.plot(timeline[80: 130], dataDynamic['DP1'][80: 130],
#          color='blue', linestyle='-', linewidth=2,
#          label='Pi Improved Model')
# plt.yaxis.set_major_formatter(yticks)
# plt.title('0.08s-0.13s Zoom In')

# ---------------y0 picture template----------------- #
# fig4 = plt.figure('y0 compare')
# plt.plot(timeline, dataDynamic['Y0'],
#          linewidth=2, linestyle='--', color='red',
#          label='Desired $\Delta M$')
# plt.plot(timeline, dataAll['Y0'],
#          linewidth=2, linestyle='--', color='red',
#          label='$\Delta M$ Original Model')
# plt.plot(timeline, dataAll['Y0'],linewidth=2,
#          label='Original Model', color='blue')
# plt.plot(timeline, dataDecoder['Y0'],
#          linewidth=2, linestyle='--', color='red',
#          label='RNN Decoder Model')
# plt.plot(timeline, data_elm['Y0'],
#          linewidth=2, linestyle='-',
#          label='elm Decoder Model')
# plt.plot(timeline, dataWeinerO['Y0'],
#          linewidth=2, linestyle='-.', color='green',
#          label='Weiner Decoder Model')
# plt.plot(timeline, dataWeinerO['Y0'],
#          linewidth=2, color='coral', linestyle='-.',
#          label='Original Model with Weiner Decoder')
# plt.xlabel('time(s)')
# plt.ylabel('$\Delta M$')
# plt.title('y hat of different decoders compare in Catching task')
# plt.title('Weiner decoder $\Delta M$ comparison of Different Model')
# plt.legend()
# plt.axes([0.41, 0.42, 0.45, 0.29])
# # plt.plot(timeline[25: 150], dataDynamic['TI'][25: 150],
# #          color='green', linestyle=':', linewidth=2,
# #          label='Target')
# plt.plot(timeline[110: 170], dataDynamic['Y0'][110: 170],
#          color='red', linestyle='--', linewidth=2,
#          label='Desired $\Delta M$')
# plt.plot(timeline[50: 150], dataDecoder['Y0'][50: 150],
#          color='blue', linestyle='-', linewidth=2,
#          label='GRU Decoder Model')
# plt.plot(timeline[110: 170], dataWeinerN['Y0'][110: 170],
#          linewidth=2, color='blue', linestyle='-',
#          label='Weiner Decoder in Improved Model')
# plt.plot(timeline[110: 170], dataWeinerO['Y0'][110: 170],
#          linewidth=2, color='coral', linestyle='-.',
#          label='Weiner Decoder in Original Model')
# plt.xlabel('Time(s)')
# plt.ylabel('Position(m)')
# plt.title('0.125s-0.15s Zoom In')
# fig4.show()