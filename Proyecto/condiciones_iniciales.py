#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 19:26:41 2024

@author: juanjo
"""
import general
import numpy as np
import matplotlib.pyplot as plt
import dicke
import rydberg
import qutip as q
import time 

inicio = time.time()
# Sacamos la matriz l1
N = 40
omega = 1
delta = -1.0*omega
v = 3.0*omega
kappa = 1*omega
params = [omega, delta, v, kappa]

H, J = rydberg.rydberg(N, params)
L = q.liouvillian(H, [J])
todoh = (L.dag()).eigenstates(sparse = False, sort = 'high', eigvals = N + 20)
vects = todoh[1]

l1 = q.Qobj(np.reshape(vects[1], (H.shape[0], H.shape[1])))

# Sacamos ahora el vector de trazas
vectorp = np.linspace(-10.0, 10.0, 45)
res = []
veces = 100
for p in vectorp:
    traza = 0.0
    print(p)
    for i in range(veces):
        d, ini = general.densidad_p(N, 0 - p, 1.0 + p)
        traza += (l1*d).tr()
    res.append(traza/veces)

final = time.time()
print(final-inicio)

# Representamos todo
plt.title(r'Tr($l_2$$\rho_0$)-p')
plt.xlabel('p')
plt.ylabel(r'Tr($l_2$$\rho_0$)')
plt.plot(vectorp, res)