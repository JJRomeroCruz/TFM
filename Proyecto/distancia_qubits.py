#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:31:31 2024

@author: juanjo
"""
import numpy as np
import sympy as sp
import scipy as sc
import matplotlib.pyplot as plt
import qutip as q
import dicke
import general
import qubits

""" Vamos a sacar la distancia de Hilbert Schmidt """
# Generamos el hamiltoniano y los operadores de salto
N = 1
sigma = 1.0
k = 0.7*sigma
#g = 0.6*(1/np.sqrt(N) + 1)*np.sqrt(2)*sigma
#g = 1.0*sigma
params = [sigma, k]
H, J = qubits.qubits(N, params)

# Matriz densidad inicial y vector inicial
d0, ini = qubits.densidad(N)
#ini = np.eye(N*N)[0]
#d0 = general.ketbra(ini, ini)

# Sacamos el lindbladiano y lo diagonalizamos
L, b = general.Limblad(H, [J])
L = np.matrix(L, dtype = complex)
# Diagonalizamos
Lq = q.Qobj(L)
Lqh = Lq.dag()
todo = Lq.eigenstates(sparse = False, sort = 'high', eigvals = 4)
todoh = Lqh.eigenstates(sparse = False, sort = 'high', eigvals = 4)
vals = todo[0]

# autoMatrices derecha
r = [np.reshape(elemento, d0.shape) for elemento in todo[1]]
#r = [np.asarray(tup[2], dtype = complex) for tup in todo]
#r = [np.squeeze(np.asarray(elemento, dtype = complex)) for elemento in r]
#r = [elemento/np.linalg.norm(elemento) for elemento in r]
#r = [np.reshape(elemento, (d0.shape[0], d0.shape[1])) for elemento in r]

# automatrices izquierda
l = [np.reshape(elemento, d0.shape) for elemento in todoh[1]]
#l = [np.asarray(tup[2], dtype = complex) for tup in todoh]
#l = [np.squeeze(np.asarray(elemento, dtype = complex)) for elemento in l]
#l = [elemento/np.linalg.norm(elemento) for elemento in l]
#l = [np.reshape(elemento, (d0.shape[0], d0.shape[1])) for elemento in l]

# Sacamos Mpemba1 y Mpemba2
U2, U_cambio = general.Mpemba2_mejorada(L, l, vals, d0, N, ini)
U1, U_cambio = general.Mpemba1_mejorada(L, l, vals, d0, N, ini)


# Sacamos Mpemba_ang
segundo_maximo, indice_segundo_maximo = general.buscar_segundo_maximo(list(np.real(vals)))
L1 = l[indice_segundo_maximo]
posibles, traza = general.buscar_angulos(L1, d0, N)
theta, phi = posibles[traza.index(min(traza))]
#theta, phi = posibles[0]
U3 = general.Mpemba_sep(theta, phi, N)
d0_exp3 = np.dot(np.dot(U3, d0), np.conjugate(U3.T))


# Sacamos la matriz inicial con cada transformacion
d0_exp1 = np.dot(np.dot(U1, d0), np.conjugate(U1.T))
d0_exp2 = np.dot(np.dot(U2, d0), np.conjugate(U2.T))
#d0_exp2 = d0_exp2/np.trace(d0_exp2)

# Calculamos la solucion
tiempo = np.linspace(0, 100, 1000)
v1 = general.solucion(d0, r, l, vals, tiempo)
v2 = general.solucion(d0_exp1, r, l, vals, tiempo)
v3 = general.solucion(d0_exp2, r, l, vals, tiempo)
v4 = general.solucion(d0_exp3, r, l, vals, tiempo)
dens = [v1, v2, v3, v4]
#dens = [v1, v2, v3]
# Sacamos el estado estacionario
#est = general.estacionario_q(H, [J])
est = r[0]
#est = general.estacionario_bueno(vals, r, l, d0)
#est = [elemento/np.trace(elemento) for elemento in est]
#est = [np.dot(U_cambio, np.dot(elemento, np.conjugate(U_cambio.T))) for elemento in est]

# Representamos la distancia de Hilbert Smichdt al estado estacionario
#m = 0
ob = [[np.sqrt(np.trace(np.dot(np.conjugate((v[i] - est).T), (v[i] - est)))) for i in range(len(v))] for v in dens]
plt.plot(tiempo, ob[0], 'b-', label = 'Random')
#plt.plot(tiempo, ob[1], 'r-', label = 'Mpemba_cero')
plt.plot(tiempo, ob[2], 'g-', label = 'Mpemba_nocero')
plt.plot(tiempo, ob[3], 'y-', label = 'Mpemba_ang')
#plt.xlim(0, 10)
#plt.ylim(0.0, 0.5)
#plt.ylim(0.0, 1.0040)
#plt.xlim(0.0, 15.0)
plt.legend(loc = 'upper right')
plt.title('Distancia de Hilbert-Schmidt. Modelo de Dicke para N = ' + str(N) + ', k = ' + str(k))
plt.show()
#ploteo_MC(ob, tiempo)