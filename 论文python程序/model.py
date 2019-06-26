import numpy as np
from keras.models import load_model
from keras.engine.topology import Layer
from keras import backend as K
from sklearn.externals import joblib
import ukf

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
#         outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]


class BMI_Model(object):
    """docstring for BMI_Model"""
    ns = 300
    delt = 1
    gold1 = 0
    gold2 = 0
    decoder = load_model('encoder_decoder_gru_python_data_145.h5', custom_objects={'AttentionLayer': AttentionLayer})
    dynamic_decoder = load_model('new_encoder_decoder_model_dynamic.h5')
    p = np.eye(1) * 0.00001

    def __init__(self, constant, initalVal):
        super(BMI_Model, self).__init__()
        self.Ix = constant['Ix']
        self.Vx = constant['Vx']
        self.Br = constant['Br']
        self.Bu = constant['Bu']
        self.row = constant['row']
        self.Thetax = constant['Thetax']
        self.etax = constant['etax']
        self.lamda1x = constant['lamda1x']
        self.lamda2x = constant['lamda2x']
        self.Gammax = constant['Gammax']
        self.Cx = constant['Cx']
        self.epsx = constant['epsx']
        self.xix = constant['xix']
        self.hx = constant['hx']
        self.TI = constant['TI']
        self.TIold = self.TI
        self.Eix = constant['Eix']
        self.delta = constant['delta']
        self.v = constant['v']
        # self.g0 = np.random.normal(0.75, 0.05)
        self.g0 = initalVal['g0']
        self.x1 = initalVal['x1']
        self.x1old = self.x1
        self.x2 = initalVal['x2']
        self.x2old = self.x2
        self.r1 = max(self.TI - self.x1 + self.Br, 0)
        self.r2 = max(1 - self.TI - self.x2 + self.Br, 0)
        self.y1 = initalVal['y1']
        self.y2 = initalVal['y2']
        self.u1 = max(self.Bu, 0)
        self.u2 = max(self.Bu, 0)
        self.g1 = initalVal['g1']
        self.p1 = initalVal['p1']
        self.dp1 = initalVal['dp1']
        self.p2 = initalVal['p2']
        self.dp2 = initalVal['dp2']
        self.gamaS1 = self.y1
        self.gamaS2 = self.y2
        self.gamaD1 = self.row * max(self.u1 - self.u2, 0)
        self.gamaD2 = self.row * max(self.u2 - self.u1, 0)
        self.q1 = initalVal['q1']
        self.q2 = initalVal['q2']
        self.f1 = initalVal['f1']
        self.f2 = initalVal['f2']
        self.a1 = self.y1 + self.q1 + self.f1
        self.a2 = self.y2 + self.q2 + self.f2
        self.s1_1 = initalVal['s1_1']
        self.s1_2 = initalVal['s1_2']
        self.s2_1 = initalVal['s2_1']
        self.s2_2 = initalVal['s2_2']
        self.alfa1 = self.a1 + self.delta * self.s1_1
        self.alfa2 = self.a2 + self.delta * self.s2_1
        self.c1 = initalVal['c1']
        self.c2 = initalVal['c2']
        self.y0 = initalVal['y0']
        self.dx1 = 0
        self.dx2 = 0
        self.v1 = 0
        self.v2 = 0
        self.vObj = initalVal['vObj']
        self.compcoff = initalVal['Z']
        self.y1List = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.y2List = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.u1List = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.u2List = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.a1List = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.a2List = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.tList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.x1List = [0] * 10
        # self.decoderData = np.zeros([4, 1, 60])
        self.decoderData = np.zeros([3, 1, 60])
        self.P = 10

    def goSignal(self, ix):
        if ix >= 5:
            gnew1 = self.gold1 + self.delt * self.epsx * (- self.gold1 + (self.Cx - self.gold1) * self.g0)
            gnew2 = self.gold2 + self.delt * self.epsx * (- self.gold2 + (self.Cx - self.gold2) * gnew1)
            self.g1 = self.g0 * gnew2 / self.Cx
            self.gold1 = gnew1
            self.gold2 = gnew2
        else:
            self.g1 = 0
        # return self

    def TPV(self, ix):
        if ix <= 50:
            self.TI += self.vObj

    def DV(self):
        self.r1 = max(self.TI - self.x1 + self.Br, 0)
        self.r2 = max(1 - self.TI - self.x2 + self.Br, 0)
        # return self

    def TVV(self):
        # self.dx1 = self.x1 - self.x1old
        # self.dx2 = self.x2 - self.x2old
        self.vObj = self.TI - self.TIold
        self.TIold = self.TI

    def RVV(self):
        self.v1 = self.delt * (self.vObj - self.dp1) * self.compcoff
        self.v2 = self.delt * (self.vObj - self.dp2) * self.compcoff

    def DVV(self):
        self.u1 = max(self.g1 * (self.r1 - self.r2) + self.Bu, 0)
        self.u2 = max(self.g1 * (self.r2 - self.r1) + self.Bu, 0)
        # return self

    def DVVDynamic(self):
        self.u1 = max(self.g1 * (self.r1 - self.r2 + self.v1 - self.v2) + self.Bu, 0)
        self.u2 = max(self.g1 * (self.r2 - self.r1 + self.v2 - self.v1) + self.Bu, 0)

    def OPV(self):
        utemp1 = max(self.u1 - self.u2, 0)
        utemp2 = max(self.u2 - self.u1, 0)
        self.y1 += self.delt * ((1 - self.y1) * (self.etax * self.x1 + utemp1)) - self.y1 * (self.etax * self.x2 + utemp2)
        self.y2 += self.delt * ((1 - self.y2) * (self.etax * self.x2 + utemp2)) - self.y2 * (self.etax * self.x1 + utemp1)
        # return self

    def gamaS(self):
        self.gamaS1 = self.y1
        self.gamaS2 = self.y2
        # return self

    def gamaD(self):
        self.gamaD1 = self.row * max(self.u1 - self.u2, 0)
        self.gamaD2 = self.row * max(self.u2 - self.u1, 0)
        # return self

    def Ia(self):
        s1_1temp = self.Thetax * max(self.gamaS1 - self.p1, 0) + max(self.gamaD1 - self.dp1, 0)
        s2_1temp = self.Thetax * max(self.gamaS2 - self.p2, 0) + max(self.gamaD2 - self.dp2, 0)
        self.s1_1 = s1_1temp / (1 + 100 * s1_1temp ** 2)
        self.s2_1 = s2_1temp / (1 + 100 * s2_1temp ** 2)

    def II(self):
        s1_2temp = self.Thetax * max(self.gamaS1 - self.p1, 0)
        s2_2temp = self.Thetax * max(self.gamaS2 - self.p2, 0)
        self.s1_2 = s1_2temp / (1 + 100 * s1_2temp ** 2)
        self.s2_2 = s2_2temp / (1 + 100 * s2_2temp ** 2)

    def IFV(self):
        self.q1 = max(self.lamda1x * (self.s1_1 - self.s1_2 - self.Gammax), 0)
        self.q2 = max(self.lamda2x * (self.s2_1 - self.s2_2 - self.Gammax), 0)

    def SFV(self):
        self.f1 += self.delt * ((1 - self.f1) * self.hx * self.s1_1 - self.xix * self.f1 * (self.f2 + self.s1_1))
        self.f2 += self.delt * ((1 - self.f2) * self.hx * self.s2_1 - self.xix * self.f2 * (self.f1 + self.s2_1))

    def OFPV(self):
        self.a1 = self.y1 + self.f1 + self.q1
        self.a2 = self.y2 + self.f2 + self.q2

    def PPV(self):
        xt1 = max(0.5 * self.y1 + self.s2_1 - self.s1_1, 0)
        xt2 = max(0.5 * self.y2 + self.s1_1 - self.s2_1, 0)
        self.x1 += self.delt * (xt1 * (1 - self.x1) - self.x1 * xt2)
        self.x2 += self.delt * (xt2 * (1 - self.x2) - self.x2 * xt1)

    def alfa(self):
        self.alfa1 = self.a1 + self.delta * self.s1_1
        self.alfa2 = self.a2 + self.delta * self.s2_1

    def CI(self):
        self.c1 += self.delt * self.v * (- self.c1 + self.alfa1)
        self.c2 += self.delt * self.v * (- self.c2 + self.alfa2)

    def Y0(self):
        self.y0 = max(self.c1 - self.p1, 0) - max(self.c2 - self.p2, 0)

    def Limb(self):
        self.dp1 += self.delt * (1 / self.Ix) * (self.y0 + self.Eix - self.Vx * self.dp1)
        self.dp2 += self.delt * (1 / self.Ix) * (- self.y0 - self.Eix - self.Vx * self.dp2)
        self.p1 += self.delt * self.dp1
        self.p2 += self.delt * self.dp2

    def modelRun(self, trainList=['U1', 'U2', 'Y1', 'Y2', 'A1', 'A2'], ix=ns):
        U1 = []
        U2 = []
        Y1 = []
        Y2 = []
        A1 = []
        A2 = []
        Y0list = []
        P1 = []
        DP1 = []
        DP2 = []
        TI = []
        R1 = []
        R2 = []
        X1 = []
        X2 = []
        Q1 = []
        F1 = []
        for i in range(ix):
            U1.append(self.u1)
            U2.append(self.u2)
            Y1.append(self.y1)
            Y2.append(self.y2)
            A1.append(self.a1)
            A2.append(self.a2)
            Y0list.append(self.y0)
            P1.append(self.p1)
            DP1.append(self.dp1)
            DP2.append(self.dp2)
            TI.append(self.TI)
            R1.append(self.r1)
            R2.append(self.r2)
            X1.append(self.x1)
            X2.append(self.x2)
            Q1.append(self.q1)
            F1.append(self.f1)
            self.goSignal(i)
            self.TPV(i)
            self.DV()
            self.DVV()
            self.OPV()
            self.gamaS()
            self.gamaD()
            self.Ia()
            self.II()
            self.IFV()
            self.SFV()
            self.OFPV()
            self.PPV()
            self.alfa()
            self.CI()
            self.Y0()
            self.Limb()
        dic = {'U1': U1, 'U2': U2,
               'Y1': Y1, 'Y2': Y2,
               'A1': A1, 'A2': A2,
               'Y0': Y0list, 'P1': P1,
               'DP1': DP1, 'TI': TI,
               'R1': R1, 'R2': R2,
               'DP2': DP2, 'X1': X1,
               'X2': X2, 'Q1': Q1,
               'F1': F1}
        return dic

    def modelRunDynamic(self, trainList=['U1', 'U2', 'Y1', 'Y2', 'A1', 'A2'], ix=ns):
            U1 = []
            U2 = []
            Y1 = []
            Y2 = []
            A1 = []
            A2 = []
            Y0list = []
            P1 = []
            DP1 = []
            DP2 = []
            TI = []
            V1 = []
            V2 = []
            R1 = []
            R2 = []
            X1 = []
            for i in range(ix):
                U1.append(self.u1)
                U2.append(self.u2)
                Y1.append(self.y1)
                Y2.append(self.y2)
                A1.append(self.a1)
                A2.append(self.a2)
                Y0list.append(self.y0)
                P1.append(self.p1)
                DP1.append(self.dp1)
                DP2.append(self.dp2)
                TI.append(self.TI)
                V1.append(self.v1)
                V2.append(self.v2)
                R1.append(self.r1)
                R2.append(self.r2)
                X1.append(self.x1)
                self.goSignal(i)
                self.TPV(i)
                self.DV()
                self.TVV()
                self.RVV()
                self.DVVDynamic()
                self.OPV()
                self.gamaS()
                self.gamaD()
                self.Ia()
                self.II()
                self.IFV()
                self.SFV()
                self.OFPV()
                self.PPV()
                self.alfa()
                self.CI()
                self.Y0()
                self.Limb()
            dic = {'U1': U1, 'U2': U2,
                   'Y1': Y1, 'Y2': Y2,
                   'A1': A1, 'A2': A2,
                   'Y0': Y0list, 'P1': P1,
                   'DP1': DP1, 'TI': TI,
                   'V1': V1, 'V2': V2,
                   'R1': R1, 'R2': R2,
                   'DP2': DP2, 'X1': X1}
            return dic

    def decoderY0(self, ix):
        stdsc = joblib.load("stdsc.m")
        temp = self.y1List + self.y2List + self.u1List + self.u2List + self.a1List + self.a2List
        temp = stdsc.transform(np.array(temp, dtype=np.float32).reshape(1, -1))
        self.y1List.pop()
        self.y1List.insert(0, self.y1)
        self.y2List.pop()
        self.y2List.insert(0, self.y2)
        self.u1List.pop()
        self.u1List.insert(0, self.u1)
        self.u2List.pop()
        self.u2List.insert(0, self.u2)
        self.a1List.pop()
        self.a1List.insert(0, self.a1)
        self.a2List.pop()
        self.a2List.insert(0, self.a2)
        if ix < 4:
            self.decoderData[ix, :, :] = np.array(temp)
            self.y0 = 0
        else:
            self.decoderData = np.vstack((self.decoderData, np.array(temp).reshape(1, 1, 60)))
            if ix == 4:
                pred = self.decoder.predict(self.decoderData, 5)
            else:
                Predarray = np.array(self.decoderData[ix - 5: ix, :, :])
                pred = self.decoder.predict(Predarray, 5)
            self.y0 = pred[4, 0, 0]

    def decoderY0_Batch4(self, ix):
        stdsc = joblib.load("stdsc.m")
        temp = self.y1List + self.y2List + self.u1List + self.u2List + self.a1List + self.a2List
        temp = stdsc.transform(np.array(temp).reshape(1, 60))
        self.y1List.pop()
        self.y1List.insert(0, self.y1)
        self.y2List.pop()
        self.y2List.insert(0, self.y2)
        self.u1List.pop()
        self.u1List.insert(0, self.u1)
        self.u2List.pop()
        self.u2List.insert(0, self.u2)
        self.a1List.pop()
        self.a1List.insert(0, self.a1)
        self.a2List.pop()
        self.a2List.insert(0, self.a2)
        if ix < 3:
            self.decoderData[ix, :, :] = np.array(temp)
            self.y0 = 0
        else:
            self.decoderData = np.vstack((self.decoderData, np.array(temp).reshape(1, 1, 60)))
            if ix == 3:
                pred = self.decoder.predict(self.decoderData, 4)
            else:
                Predarray = np.array(self.decoderData[ix - 4: ix, :, :])
                pred = self.decoder.predict(Predarray, 4)
            self.y0 = pred[3, 0, 0]

    def decoderY0_Batch0(self, ix):
        stdsc = joblib.load("stdsc_python.m")
        temp = self.u1List + self.u2List + self.y1List + self.y2List + self.a1List + self.a2List
        temp = stdsc.transform(np.array(temp).reshape(1, -1).astype(np.float))
        t_temp = self.tList
        self.y1List.pop(0)
        self.y1List.insert(9, self.y1)
        self.y2List.pop(0)
        self.y2List.insert(9, self.y2)
        self.u1List.pop(0)
        self.u1List.insert(9, self.u1)
        self.u2List.pop(0)
        self.u2List.insert(9, self.u2)
        self.a1List.pop(0)
        self.a1List.insert(9, self.a1)
        self.a2List.pop(0)
        self.a2List.insert(9, self.a2)
        self.tList.pop(0)
        self.tList.insert(9, self.y0)
        temp = np.array(temp).reshape(1, 1, 60)
        t_temp = np.array(t_temp).reshape(1, 1, 10)
        pred = self.decoder.predict([temp, t_temp])
        self.y0 = pred[0, 0]

    def decoderY0_Dynamic(self, ix):
        stdsc = joblib.load("stdsc_python_dynamic.m")
        temp = self.u1List + self.u2List + self.y1List + self.y2List + self.a1List + self.a2List
        temp = stdsc.transform(np.array(temp).reshape(1, -1).astype(np.float))
        t_temp = self.tList
        self.y1List.pop(0)
        self.y1List.insert(9, self.y1)
        self.y2List.pop(0)
        self.y2List.insert(9, self.y2)
        self.u1List.pop(0)
        self.u1List.insert(9, self.u1)
        self.u2List.pop(0)
        self.u2List.insert(9, self.u2)
        self.a1List.pop(0)
        self.a1List.insert(9, self.a1)
        self.a2List.pop(0)
        self.a2List.insert(9, self.a2)
        self.tList.pop(0)
        self.tList.insert(9, self.y0)
        temp = np.array(temp).reshape(1, 1, 60)
        t_temp = np.array(t_temp).reshape(1, 1, 10)
        pred = self.dynamic_decoder.predict([temp, t_temp])
        self.y0 = pred[0, 0, 0]

    def mlp_x1(self, model, stdc):
        tmp = np.array([self.y1, self.y2, self.u1, self.u2, self.dp1, self.dp2,
                        self.p1, self.p2])
        tmp = stdc.transform(tmp.reshape(1, 8))
        self.x1 = model.predict(tmp)[0][0]
        self.x2 = 1 - self.x1

    def mlp_x_to_x(self, model, stdc):
        tmp = stdc.transform(self.x1)[0][0]
        self.x1 = model.predict(np.array(tmp).reshape(-1,1))[0][0]
        self.x2 = 1 - self.x1
    
    def mlp_x1_5fx1(self, model, stdc):
        tmp = np.array([self.y1, self.y2, self.u1, self.u2, self.p1, self.p2, self.dp1, self.dp2])
        tmp = stdc.transform(tmp.reshape(1, 8))
        self.x1 = model.predict(tmp)[0][0]
        self.x2 = 1 - self.x1
    
    def mlp_x1_8fx1_plus_x1(self, model, stdc):
        tmp = np.array([self.y1, self.y2, self.u1, self.u2, self.p1, self.p2, self.dp1, self.dp2, self.x1])
        tmp = stdc.transform(tmp.reshape(1, 9))
        self.x1 = model.predict(tmp)[0][0]
        self.x2 = 1 - self.x1


    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def decoder_Weiner(self, W):
        self.y1List.pop()
        self.y1List.insert(0, self.y1)
        self.y2List.pop()
        self.y2List.insert(0, self.y2)
        self.u1List.pop()
        self.u1List.insert(0, self.u1)
        self.u2List.pop()
        self.u2List.insert(0, self.u2)
        self.a1List.pop()
        self.a1List.insert(0, self.a1)
        self.a2List.pop()
        self.a2List.insert(0, self.a2)
        temp = self.y1List + self.y2List + self.u1List + self.u2List + self.a1List + self.a2List
        temp = np.array(temp).reshape(1, 60)
        Yhat = np.dot(temp, W)
        self.y0 = Yhat[0][0]

    def decoder_Kalman(self, A, C, Q, R):
        temp = [self.y1, self.y2, self.u1, self.u2, self.a1, self.a2]
        temp = np.array(temp).reshape(6, 1)
        Yhat = A * self.y0
        self.P = A * self.P * A + R
        K_temp = C * self.P * C.T + Q
        K = (self.P * C.T) * np.array(K_temp).I
        Yhat += np.array(K) * (temp - C.reshape(6, 1) * Yhat)
        self.y0 = np.array(Yhat)[0][0]
        Phat = (1 - np.array(K) * C) * self.P
        self.P = np.array(Phat)[0][0]

    def modelRunDecoder(self, trainList=['U1', 'U2', 'Y1', 'Y2', 'A1', 'A2'], ix=ns):
        U1 = []
        U2 = []
        Y1 = []
        Y2 = []
        A1 = []
        A2 = []
        Y0list = []
        P1 = []
        DP1 = []
        TI = []
        X1 = []
        R1 = []
        for i in range(ix):
            U1.append(self.u1)
            U2.append(self.u2)
            Y1.append(self.y1)
            Y2.append(self.y2)
            A1.append(self.a1)
            A2.append(self.a2)
            Y0list.append(self.y0)
            P1.append(self.p1)
            DP1.append(self.dp1)
            TI.append(self.TI)
            X1.append(self.x1)
            R1.append(self.r1)
            self.goSignal(i)
            self.TPV(i)
            self.DV()
            self.DVV()
            self.OPV()
            self.gamaS()
            self.gamaD()
            self.Ia()
            self.II()
            self.IFV()
            self.SFV()
            self.OFPV()
            self.PPV()
            self.decoderY0_Batch0(i)
            self.Limb()
        dic = {'U1': U1, 'U2': U2,
               'Y1': Y1, 'Y2': Y2,
               'A1': A1, 'A2': A2,
               'Y0': Y0list, 'P1': P1,
               'DP1': DP1, 'TI': TI,
               'X1': X1, 'R1': R1}
        return dic

    def modelRun_Weiner(self, W, trainList=['U1', 'U2', 'Y1', 'Y2', 'A1', 'A2'], ix=ns):
        U1 = []
        U2 = []
        Y1 = []
        Y2 = []
        A1 = []
        A2 = []
        Y0list = []
        P1 = []
        DP1 = []
        TI = []
        R1 = []
        X1 = []
        for i in range(ix):
            U1.append(self.u1)
            U2.append(self.u2)
            Y1.append(self.y1)
            Y2.append(self.y2)
            A1.append(self.a1)
            A2.append(self.a2)
            Y0list.append(self.y0)
            P1.append(self.p1)
            DP1.append(self.dp1)
            TI.append(self.TI)
            R1.append(self.r1)
            X1.append(self.x1)
            self.goSignal(i)
            self.TPV(i)
            self.DV()
            self.TVV()
            self.RVV()
            self.DVVDynamic()
            # self.DVV()
            self.OPV()
            self.gamaS()
            self.gamaD()
            self.Ia()
            self.II()
            self.IFV()
            self.SFV()
            self.OFPV()
            self.PPV()
            # self.alfa()
            # self.CI()
            # self.Y0()
            # self.decoderY0(i)
            # self.decoderY0_Batch4(i)
            self.decoder_Weiner(W)
            self.Limb()
        dic = {'U1': U1, 'U2': U2,
               'Y1': Y1, 'Y2': Y2,
               'A1': A1, 'A2': A2,
               'Y0': Y0list, 'P1': P1,
               'DP1': DP1, 'TI': TI,
               'R1': R1, 'X1': X1}
        # for key in trainList:
        #     dic[key] = self.initalZero(dic[key])
        #     dic[key] = self.inital(dic[key])
        return dic

    def modelRun_Kalman(self, A, C, Q, R, trainList=['U1', 'U2', 'Y1', 'Y2', 'A1', 'A2'], ix=ns):
        U1 = []
        U2 = []
        Y1 = []
        Y2 = []
        A1 = []
        A2 = []
        Y0list = []
        P1 = []
        DP1 = []
        for i in range(ix):
            U1.append(self.u1)
            U2.append(self.u2)
            Y1.append(self.y1)
            Y2.append(self.y2)
            A1.append(self.a1)
            A2.append(self.a2)
            Y0list.append(self.y0)
            P1.append(self.p1)
            DP1.append(self.dp1)
            self.goSignal(i)
            self.TPV(i)
            self.DV()
            self.TVV()
            self.RVV()
            self.DVVDynamic()
            # self.DVV()
            self.OPV()
            self.gamaS()
            self.gamaD()
            self.Ia()
            self.II()
            self.IFV()
            self.SFV()
            self.OFPV()
            self.PPV()
            self.decoder_Kalman(A, C, Q, R)
            self.Limb()
        dic = {'U1': U1, 'U2': U2,
               'Y1': Y1, 'Y2': Y2,
               'A1': A1, 'A2': A2,
               'Y0': Y0list, 'P1': P1,
               'DP1': DP1}
        return dic

    def modelRun_WeinerOriginal(self, W, trainList=['U1', 'U2', 'Y1', 'Y2', 'A1', 'A2'], ix=ns):
        U1 = []
        U2 = []
        Y1 = []
        Y2 = []
        A1 = []
        A2 = []
        Y0list = []
        P1 = []
        DP1 = []
        TI = []
        R1 = []
        X1 = []
        for i in range(ix):
            U1.append(self.u1)
            U2.append(self.u2)
            Y1.append(self.y1)
            Y2.append(self.y2)
            A1.append(self.a1)
            A2.append(self.a2)
            Y0list.append(self.y0)
            P1.append(self.p1)
            DP1.append(self.dp1)
            TI.append(self.TI)
            R1.append(self.r1)
            X1.append(self.x1)
            self.goSignal(i)
            self.TPV(i)
            self.DV()
            # self.TVV()
            # self.RVV()
            # self.DVVDynamic()
            self.DVV()
            self.OPV()
            self.gamaS()
            self.gamaD()
            self.Ia()
            self.II()
            self.IFV()
            self.SFV()
            self.OFPV()
            self.PPV()
            # self.alfa()
            # self.CI()
            # self.Y0()
            # self.decoderY0(i)
            # self.decoderY0_Batch4(i)
            self.decoder_Weiner(W)
            self.Limb()
        dic = {'U1': U1, 'U2': U2,
               'Y1': Y1, 'Y2': Y2,
               'A1': A1, 'A2': A2,
               'Y0': Y0list, 'P1': P1,
               'DP1': DP1, 'TI': TI,
               'X1': X1, 'R1': R1}
        # for key in trainList:
        #     dic[key] = self.initalZero(dic[key])
            # dic[key] = self.inital(dic[key])
        return dic

    def modelRun_mlp(self, model, stdc, trainList=['U1', 'U2', 'Y1', 'Y2', 'A1', 'A2'], ix=ns):
        U1 = []
        U2 = []
        Y1 = []
        Y2 = []
        A1 = []
        A2 = []
        Y0list = []
        P1 = []
        DP1 = []
        DP2 = []
        TI = []
        R1 = []
        R2 = []
        X1 = []
        X2 = []
        for i in range(ix):
            U1.append(self.u1)
            U2.append(self.u2)
            Y1.append(self.y1)
            Y2.append(self.y2)
            A1.append(self.a1)
            A2.append(self.a2)
            Y0list.append(self.y0)
            P1.append(self.p1)
            DP1.append(self.dp1)
            DP2.append(self.dp2)
            TI.append(self.TI)
            R1.append(self.r1)
            R2.append(self.r2)
            X1.append(self.x1)
            X2.append(self.x2)
            self.goSignal(i)
            self.TPV(i)
            self.DV()
            self.DVV()
            self.OPV()
            self.gamaS()
            self.gamaD()
            self.Ia()
            self.II()
            self.IFV()
            self.SFV()
            self.OFPV()
            self.mlp_x1_8fx1_plus_x1(model, stdc)
            # self.alfa()
            # self.CI()
            self.decoderY0_Dynamic(i)
            # self.Y0()
            self.Limb()
            x_data = np.array([self.y1, self.y2, self.u1, self.u2, self.p1, self.p2, self.dp1, self.dp2, self.x1]).reshape(1, -1)
            xc, self.p = ukf.UKF(model=model, stdc=stdc, yc=self.x1, n=1, a=0.9, k=0, b=1, x0=x_data, p=self.p)
            self.x1 = xc[0]
            self.x2 = 1 - self.x1
        dic = {'U1': U1, 'U2': U2,
               'Y1': Y1, 'Y2': Y2,
               'A1': A1, 'A2': A2,
               'Y0': Y0list, 'P1': P1,
               'DP1': DP1, 'TI': TI,
               'R1': R1, 'R2': R2,
               'DP2': DP2, 'X1': X1,
               'X2': X2}
        return dic


    def modelRun_mlp_5fx1(self, model, stdc, trainList=['U1', 'U2', 'Y1', 'Y2', 'A1', 'A2'], ix=ns):
        U1 = []
        U2 = []
        Y1 = []
        Y2 = []
        A1 = []
        A2 = []
        Y0list = []
        P1 = []
        DP1 = []
        DP2 = []
        TI = []
        R1 = []
        R2 = []
        X1 = []
        X2 = []
        for i in range(ix):
            U1.append(self.u1)
            U2.append(self.u2)
            Y1.append(self.y1)
            Y2.append(self.y2)
            A1.append(self.a1)
            A2.append(self.a2)
            Y0list.append(self.y0)
            P1.append(self.p1)
            DP1.append(self.dp1)
            DP2.append(self.dp2)
            TI.append(self.TI)
            R1.append(self.r1)
            R2.append(self.r2)
            X1.append(self.x1)
            X2.append(self.x2)
            self.goSignal(i)
            self.TPV(i)
            self.DV()
            self.DVV()
            self.OPV()
            self.gamaS()
            self.gamaD()
            self.Ia()
            self.II()
            self.IFV()
            self.SFV()
            self.OFPV()
            self.mlp_x1_8fx1_plus_x1(model, stdc)
            self.alfa()
            self.CI()
            self.Y0()
            self.Limb()
        dic = {'U1': U1, 'U2': U2,
               'Y1': Y1, 'Y2': Y2,
               'A1': A1, 'A2': A2,
               'Y0': Y0list, 'P1': P1,
               'DP1': DP1, 'TI': TI,
               'R1': R1, 'R2': R2,
               'DP2': DP2, 'X1': X1,
               'X2': X2}
        return dic

    def modelRunDecoderDynamic(self, trainList=['U1', 'U2', 'Y1', 'Y2', 'A1', 'A2'], ix=ns):
        U1 = []
        U2 = []
        Y1 = []
        Y2 = []
        A1 = []
        A2 = []
        Y0list = []
        P1 = []
        DP1 = []
        TI = []
        X1 = []
        R1 = []
        for i in range(ix):
            U1.append(self.u1)
            U2.append(self.u2)
            Y1.append(self.y1)
            Y2.append(self.y2)
            A1.append(self.a1)
            A2.append(self.a2)
            Y0list.append(self.y0)
            P1.append(self.p1)
            DP1.append(self.dp1)
            TI.append(self.TI)
            X1.append(self.x1)
            R1.append(self.r1)
            self.goSignal(i)
            self.TPV(i)
            self.DV()
            self.TVV()
            self.RVV()
            self.DVVDynamic()
            self.OPV()
            self.gamaS()
            self.gamaD()
            self.Ia()
            self.II()
            self.IFV()
            self.SFV()
            self.OFPV()
            self.PPV()
            self.decoderY0_Dynamic(i)
            self.Limb()
        dic = {'U1': U1, 'U2': U2,
               'Y1': Y1, 'Y2': Y2,
               'A1': A1, 'A2': A2,
               'Y0': Y0list, 'P1': P1,
               'DP1': DP1, 'TI': TI,
               'X1': X1, 'R1': R1}
        return dic

    def modelRun_mlp_x_to_x(self, model, stdc, trainList=['U1', 'U2', 'Y1', 'Y2', 'A1', 'A2'], ix=ns):
        U1 = []
        U2 = []
        Y1 = []
        Y2 = []
        A1 = []
        A2 = []
        Y0list = []
        P1 = []
        DP1 = []
        DP2 = []
        TI = []
        R1 = []
        R2 = []
        X1 = []
        X2 = []
        for i in range(ix):
            U1.append(self.u1)
            U2.append(self.u2)
            Y1.append(self.y1)
            Y2.append(self.y2)
            A1.append(self.a1)
            A2.append(self.a2)
            Y0list.append(self.y0)
            P1.append(self.p1)
            DP1.append(self.dp1)
            DP2.append(self.dp2)
            TI.append(self.TI)
            R1.append(self.r1)
            R2.append(self.r2)
            X1.append(self.x1)
            X2.append(self.x2)
            self.goSignal(i)
            self.TPV(i)
            self.DV()
            self.DVV()
            self.OPV()
            self.gamaS()
            self.gamaD()
            self.Ia()
            self.II()
            self.IFV()
            self.SFV()
            self.OFPV()
            self.mlp_x_to_x(model, stdc)
            self.alfa()
            self.CI()
            self.Y0()
            self.Limb()
        dic = {'U1': U1, 'U2': U2,
               'Y1': Y1, 'Y2': Y2,
               'A1': A1, 'A2': A2,
               'Y0': Y0list, 'P1': P1,
               'DP1': DP1, 'TI': TI,
               'R1': R1, 'R2': R2,
               'DP2': DP2, 'X1': X1,
               'X2': X2}
        return dic
