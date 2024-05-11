#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:14:24 2024

@author: juanjo
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import rydberg
import general

""" Vamos a sacar la distancia de Hilbert Schmidt """
# Generamos el hamiltoniano y los operadores de salto
N = 2
sigma = 1.0
delta = 0.5*sigma
v = 3.0*sigma
k = 2.0*sigma
params = [sigma, delta, v, k]
H, J = rydberg.rydberg(N, params)

# Matriz densidad inicial y vector inicial
d0, ini = rydberg.densidad_rydberg(N, H)

# Sacamos el lindbladiano y lo diagonalizamos
L, b = general.Limblad(H, [J])
todo = L.eigenvects()
todoh = (L.H).eigenvects()
vals = np.array([tup[0] for tup in todo], dtype = complex)
print(type(vals[0]))
r = [np.asarray(tup[2], dtype = complex) for tup in todo]
r = [np.squeeze(np.asarray(elemento, dtype = complex)) for elemento in r]
r = [elemento/np.linalg.norm(elemento) for elemento in r]
r = [np.reshape(elemento, (d0.shape[0], d0.shape[1])) for elemento in r]

l = [np.asarray(tup[2], dtype = complex) for tup in todoh]
l = [np.squeeze(np.asarray(elemento, dtype = complex)) for elemento in l]
l = [elemento/np.linalg.norm(elemento) for elemento in l]
l = [np.reshape(elemento, (d0.shape[0], d0.shape[1])) for elemento in l]

# Sacamos Mpemba1. Mpemba2
U2, U_cambio = general.Mpemba2_mejorada(L, l, vals, d0, N, ini)
U1, U_cambio = general.Mpemba1_mejorada(L, l, vals, d0, N, ini)

d0_exp1 = np.dot(np.dot(U1, d0), np.conjugate(U1.T))
d0_exp2 = np.dot(np.dot(U2, d0), np.conjugate(U2.T))
d0_exp2 = d0_exp2/np.trace(d0_exp2)
d0_exp1 = d0_exp1/np.trace(d0_exp1)


# Calculamos la solucion
tiempo = np.linspace(0, 50, 1000)
v1 = general.solucion(d0, r, l, vals, tiempo)
v2 = general.solucion(d0_exp1, r, l, vals, tiempo)
v3 = general.solucion(d0_exp2, r, l, vals, tiempo)

dens = [v1, v2, v3]

# Sacamos el estado estacionario
est = general.estacionario_bueno(vals, r, d0)
print(" estacionario: " + str(len(est)))
# Representamos la distancia de Hilbert Smichdt al estado estacionario
m = 0
ob = [[np.sqrt(np.trace(np.dot(np.conjugate((v[i]-est[0]).T), v[i]-est[0]))) for i in range(len(v))] for v in dens]
#ob = [[np.sqrt(np.trace(np.dot(np.conjugate((v[i] - est[m]).T), (v[i] - est[m])))) for i in range(len(v))] for v in dens]
plt.plot(tiempo, ob[0], 'b-', label = 'Random')
plt.plot(tiempo, ob[1], 'r-', label = 'Mpemba_cero')
plt.plot(tiempo, ob[2], 'g-', label = 'Mpemba_nocero')
#plt.plot(tiempo, ob[3], 'y-', label = 'Mpemba_nocero')
plt.xlim(0, 20)
#plt.ylim(0.0, 0.5)
#plt.ylim(0.0, 1.0040)
#plt.xlim(0.0, 15.0)
plt.legend(loc = 'upper right')
plt.title('Distancia de Hilbert-Schmidt. Modelo de atomos Rydberg para N = ' + str(N) + ', k = ' + str(k) + ' y delta = ' + str(delta))
plt.show()
#ploteo_MC(ob, tiempo)