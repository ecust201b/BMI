from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

# model = load_model('mlp_x1_to_4.h5')
data = pd.read_csv('data_python.csv')
P2 = 1 - data['P1']
data['P2'] = P2
train_x1 = data[['Y1', 'Y2', 'U1', 'U2']].values.reshape(233600, 4)
train_x2 = data[['P1', 'P2', 'DP1', 'DP2', 'X1']].values.reshape(233600, 5)
train_y = data['X1'].values.reshape(233600, 1)
# index1 = [i * 146 for i in range(1600)]
# index2 = [(i + 1) * 146 - 1 for i in range(1600)]
# train_x1 = np.delete(train_x1, index2, axis=0)
# train_x2 = np.delete(train_x2, index2, axis=0)
train_x = np.hstack((train_x1, train_x2))
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
train_x = stdsc.fit_transform(train_x)
joblib.dump(stdsc, "stdsc_mlp_5_to_x1_plus_x1_233600.m")
# train_y = np.delete(train_y, index1, axis=0)
X_train = train_x[:186880, :]
X_test = train_x[186880:, :]
Y_train = train_y[:186880, :]
Y_test = train_y[186880:, :]

model = Sequential()
model.add(Dense(50, input_shape=(9, )))
model.add(Activation('sigmoid'))
# model.add(Dropout(0.05))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, Y_train, batch_size=145, epochs=7,
                    validation_data=(X_test, Y_test))

model.save('model_mlp_5_to_x1_plus_x1_233600.h5')

pred = model.predict(X_test)
# f, ax = plt.subplots(figsize=(9,6), ncols=2, nrows=2)

# ax[0][0].plot(pred[:730, 0], linewidth=2, label='ANN',  color='blue')
# ax[0][0].plot(Y_test[:730, 0], linewidth=2, label='Origin', color='red', linestyle='--')
# ax[0][0].set_ylabel('$y_{i}$', fontsize=14)
# ax[0][0].legend(fontsize=12, loc=7)

# ax[0][1].plot(pred[:730, 1], linewidth=2, label='ANN',  color='blue')
# ax[0][1].plot(Y_test[:730, 1], linewidth=2, label='Origin', color='red', linestyle='--')
# ax[0][1].set_ylabel('$u_{i}$', fontsize=14)
# ax[0][1].legend(fontsize=12, loc=1)

# ax[1][0].plot(pred[:730, 2], linewidth=2, label='ANN',  color='blue')
# ax[1][0].plot(Y_test[:730, 2], linewidth=2, label='Origin', color='red', linestyle='--')
# ax[1][0].set_ylabel('$r_{i}$', fontsize=14)
# ax[1][0].legend(fontsize=12, loc=1)

# ax[1][1].plot(pred[:730, 3], linewidth=2, label='ANN',  color='blue')
# ax[1][1].plot(Y_test[:730, 3], linewidth=2, label='Origin', color='red', linestyle='--')
# ax[1][1].set_ylabel('$p_{i}$', fontsize=14)
# ax[1][1].legend(fontsize=12, loc=7)

# f.show()
# plt.show()

plt.plot(pred[:730, 0], linewidth=2, label='ANN', color='blue')
plt.plot(Y_test[:730, 0], linewidth=2, label='Origin', color='red', linestyle='--')
plt.ylabel('$\hat{x}_{i}$', fontsize=12)
plt.legend(fontsize=12, loc=1)
plt.show()
pred = model.predict(train_x)
# V = np.dot((pred - train_y).T, (pred - train_y)) / train_y.shape[0]
# print(V)
