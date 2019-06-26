from keras.models import load_model
from keras.engine.topology import Layer
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pandas as pd
import math

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

model = load_model('encoder_decoder_gru_python_data_145.h5', custom_objects={'AttentionLayer': AttentionLayer})
stdsc = joblib.load('stdsc_python.m')
df = pd.read_csv('data_python.csv')
T2 = df['Y0']
T_new = []
for i in range(1600):
    tmp = [0] * 9
    tmp.extend(T2[i * 146: (i + 1)* 146])
    for j in range(len(tmp) - 10):
        T_new.append(tmp[j: j + 10])
T_new = np.array(T_new)
train_data = np.load('python_train_data_gru_145.csv.npy')
target = np.load('python_train_target_gru_145.csv.npy')
T = target
P = train_data
P = stdsc.transform(P)
P_T = T_new
train_num = int(P.shape[0] * 0.8)
test_num = int(P.shape[0] * 0.2)
P_Train = P[:train_num, :]
T_Train = T[:train_num, :]
P_Test = P[train_num:, :]
T_Test = T[train_num:, :]
P_Train = P_Train.reshape(train_num,1,60)
P_Test = P_Test.reshape(test_num,1,60)
T_new_Train = P_T[:train_num, :]
T_new_Test = P_T[train_num:, :]
T_new_Train = T_new_Train.reshape(train_num,1,10)
T_new_Test = T_new_Test.reshape(test_num,1,10)

pred = model.predict([P_Train, T_new_Train])
plt.plot(pred[0:730], linewidth=2, label='RNN decoder', color='blue')
plt.plot(T_Train[0:730], linewidth=2, label='Original model', color='red', linestyle='--')
plt.xlabel('step')
plt.ylabel('$\Delta M$')
plt.legend(loc='best')
plt.show()
e1 = np.abs(pred[0:730] - T_Train[0:730])
sum_error1 = [sum(e1[:i]) for i in range(730)]

print(mse(T_Train[0:730], pred[0:730]),
      COR(T_Train[0:730], pred[0:730]),
      rSquared(T_Train[0:730], pred[0:730]))

train_data = np.load('python_train_data_gru_145.csv.npy').reshape(145 * 1600, 60).T
w = np.load('weiner_w.npy')
pred = []
real = []
for i in range(730):
    pred.append(np.dot(w.T, train_data[:, i]))
    real.append(T_Train[i, 0])
print(mse(real[0:730], pred[0:730]),
      COR(real[0:730], pred[0:730]),
      rSquared(real[0:730], pred[0:730]))
plt.plot(pred, linewidth=2, label='Weiner decoder', color='blue')
plt.plot(real, linewidth=2, label='Original model', color='red', linestyle='--')
plt.xlabel('step')
plt.ylabel('$\Delta M$')
plt.legend()
plt.show()
e2 = [abs(pred[i] - real[i]) for i in range(730)]
sum_error2 = [sum(e2[:i]) for i in range(730)]
plt.plot(sum_error2, linewidth=2, label='Weiner decoder', color='blue')
plt.plot(sum_error1, linewidth=2, label='RNN decoder', color='red', linestyle='--')
plt.xlabel('step')
plt.ylabel('error')
plt.legend()
plt.show()
