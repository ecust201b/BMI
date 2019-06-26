import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as sio


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

train_data = np.load('python_train_data_gru_145.csv.npy').reshape(145 * 1600, 60).T
target = np.load('python_train_target_gru_145.csv.npy').reshape(145 * 1600, 1)

znorm = 0
w = np.zeros((60,1))
e_list = []
for k in range(1600):
    for i in range(145 * k, 145 * (k + 1)):
        p = train_data[:, i].reshape(60, 1)
        deltaM = np.dot(w.T, p)
        e = target[i, 0] - deltaM[0][0]
        # for j in range(60):
        #     znorm += p[j, 0] * p[j, 0]
        # znorm = math.sqrt(znorm)
        znorm = np.linalg.norm(p)
        w = w +  np.array(0.01 * e * p / (1 + znorm)).reshape(60, 1)
        znorm = 0



np.save('weiner_w', w)
pred = []
real = []
for i in range(730):
    pred.append(np.dot(w.T, train_data[:, i]))
    real.append(target[i, 0])
# plt.figure(figsize=(10,5))
print(mse(real[0:730], pred[0:730]),
      COR(real[0:730], pred[0:730]),
      rSquared(real[0:730], pred[0:730]))
plt.plot(pred, linewidth=2, label='Weiner decoder', color='blue')
plt.plot(real, linewidth=2, label='Original model', color='red', linestyle='--')
plt.xlabel('step')
plt.ylabel('$\Delta M$')
plt.legend()
plt.show()


