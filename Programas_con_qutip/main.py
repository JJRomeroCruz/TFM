#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:33:00 2024

@author: juanjo
"""
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import qutip as q
#from qutip import eigenstates
import dicke
import general
import time

inicio = time.time()
# hamiltoniano y operadores de salto
N = 5 # el limite esta en 13 incluido, a partir de 14 peta
omega = 1.0
w = 10.0*omega
g = 20.0*omega
k = 20.0*omega
params = [omega, w, g, k]
H, J = dicke.dicke(N, params)

# lindbladiano
L = general.limblad(H, [J])

# 3 primeros autoestados
#evals, evecs = L.eigenstates(sparse = False, sort = 'high', eigvals = 10, tol = 1e-7, maxiter = 1000000)
#y = [np.imag(elemento) for elemento in evals]
#x = [np.real(elemento) for elemento in evals]
#plt.plot(x, y, 'o')
#print(evals)

# Evolucion temporal
#d0 = q.rand_dm(L.shape[0])
d0 = q.rand_dm(N)
t = np.linspace(0, 20, 100)

output = q.mesolve(H, d0, t, c_ops = [J], options = q.Options(nsteps=10000))
#output = q.mesolve(L, d0, t, c_ops = [L], options = q.Options(nsteps=10000))
evolucion = output.states

for i, dt in enumerate(evolucion):
    plt.imshow(dt.full(), extent=[0, 1, 0, 1], cmap='jet', vmin=0, vmax=1)
    plt.title(f'Tiempo = {t[i]}')
    plt.xlabel('Índice de fila')
    plt.ylabel('Índice de columna')
    plt.colorbar(label='Valor de la matriz densidad')
    plt.show()


fin = time.time()
print('Tiempo: ' + str(fin-inicio))