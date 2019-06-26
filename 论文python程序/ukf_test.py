from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pandas as pd
import ukf

def mse(target, output):
    sqsum = 0
    for x, y in zip(target, output):
        sqsum += (x - y) ** 2
    return sqsum / (len(target))

data = pd.read_csv('data_python.csv')
P2 = 1 - data['P1']
data['P2'] = P2
train_x1 = data[['Y1', 'Y2', 'U1', 'U2']].values.reshape(233600, 4)
train_x2 = data[['P1', 'P2', 'DP1', 'DP2', 'X1']].values.reshape(233600, 5)
train_y = data['X1'].values.reshape(233600, 1)
index1 = [i * 146 for i in range(1600)]
index2 = [(i + 1) * 146 - 1 for i in range(1600)]
train_x1 = np.delete(train_x1, index2, axis=0)
train_x2 = np.delete(train_x2, index2, axis=0)
train_x = np.hstack((train_x1, train_x2))
train_y = np.delete(train_y, index1, axis=0)
X_train = train_x[:186880, :]
X_test = train_x[186880:, :]
Y_train = train_y[:186880, :]
Y_test = train_y[186880:, :]
model = load_model('model_mlp_5_to_x1_plus_x1.h5')
stdc = joblib.load('stdsc_mlp_5_to_x1_plus_x1.m')
err = np.load('err.npy')
p = np.eye(1) * 0.00001
xc_list = []
real = []
y_ann = []
p000 = data['P1'].values.reshape(-1, 1)

for i in range(730):
    x_data = X_train[i, :].reshape(1, 9)
    tmp = stdc.transform(x_data)
    yc = model.predict(tmp) + err[i % 145, 0]
    y_ann.append(yc[0][0])
    xc, p = ukf.UKF(model=model, stdc=stdc, yc=p000[i, 0], n=1, a=1, k=0, b=1, x0=x_data, p=p)
    xc_list.append(xc[0])
    real.append(Y_train[i, :])

abs_err = [abs(xc_list[i] - real[i]) for i in range(len(xc_list))]
mse = mse(real, xc_list)
print(sum(abs_err), mse)

plt.plot(xc_list, linewidth=2, label='ANN with UKF', color='blue')
plt.plot(real, linewidth=2, label='Origin', color='red', linestyle='--')
plt.plot(y_ann, linewidth=2, label='ANN', color='green', linestyle='-.')
plt.ylabel('$\hat{x}_{i}$', fontsize=12)
plt.legend(fontsize=12, loc=1)
plt.show()
