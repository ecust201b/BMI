import numpy as np
import math
    
def UKF(model, stdc, yc, n=None, a=None, k=0, b=1, x0=None, p=None, Q=None):
    lamda = a ** 2 * (n + k) - n
    nc = n + lamda
    Wm = np.zeros([1, 2 * n + 1]) + 0.5 * lamda / nc
    Wm[0, 0] = lamda / nc
    Wc = Wm
    Wc[0, 0] += (1 - a ** 2 + b)
    ns = math.sqrt(nc)
    sxk = 0
    spk = 0
    syk = 0
    pyy = 0
    pxy = 0
    p = p
    x = np.array(x0[0, -1])
    #-------sigma point------#
    u, s, v =np.linalg.svd(p, full_matrices=True)
    smat = np.zeros((u.shape[0], v.shape[0]))
    smat[:u.shape[0], :u.shape[0]] = np.diag(s)
    sigma_list = np.zeros([n, 2 * n + 1])
    sigma_list[:, 0] = x
    pk = np.dot(u, np.sqrt(ns * smat))
    for k in range(2 * n):
        if k < n:
            sigma_list[:, k + 1] = (x + pk[:, k])
        else:
            sigma_list[:, k + 1] = (x - pk[:, k - n])
    sigma_list = sigma_list.T
    #------pred x ----------#
    for ks in range(2 * n + 1):
        tmp = x0[:]
        tmp[0, 8] = sigma_list[ks, :][0]
        tmp = stdc.transform(tmp)
        sigma_list[ks, :] = model.predict(tmp)[0]
        sxk += Wm[0, ks] * sigma_list[ks, :]
    #-------- pk -----------#
    for kp in range(2 * n + 1):
        spk += Wc[0, kp] * (sigma_list[kp, :] - sxk) * (sigma_list[kp, :] - sxk).T
    #--------pred y --------#
    for ky in range(2 * n + 1):
        syk += Wm[0, kp] * sigma_list[ky, :]
    #-------update----------#
    for kpy in range(2 * n + 1):
        pyy += Wc[0, kpy] * (sigma_list[kpy, :] - syk) * (sigma_list[kpy, :] - syk).T
    for kxy in range(2 * n + 1):
        pxy += Wc[0, kxy] * (sigma_list[kxy, :] - sxk) * (sigma_list[kxy, :] - syk).T
    kgs = pxy / pyy
    xc = sxk + kgs * (yc - syk)
    p = np.array(spk - kgs * pyy * kgs.T).reshape(1, 1)
    return xc, p
