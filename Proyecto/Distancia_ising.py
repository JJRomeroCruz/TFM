#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:32:29 2024

@author: juanjo
"""
import numpy as np
import matplotlib.pyplot as plt
import Prueba_Ising
import general

""" Vamos a sacar la distancia de Hilbert Schmidt """
# Generamos el hamiltoniano y los operadores de salto
N = 1
omega = 1.0
V = 1.0*omega
k = 0.1*omega
params = [sigma, w, k, g]
H, J = dicke.dicke(N, params)

# Matriz densidad inicial y vector inicial
d0, ini = dicke.densidad(N)

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

# Sacamos Mpemba_ang
segundo_maximo, indice_segundo_maximo = general.buscar_segundo_maximo(list(np.real(vals)))
L1 = l[indice_segundo_maximo]
posibles, traza = general.buscar_angulos(L1, d0, N)
theta, phi = posibles[traza.index(min(traza))]
U3 = general.Mpemba_sep(theta, phi, N)
d0_exp3 = np.dot(np.dot(U3, d0), np.conjugate(U3.T))



d0_exp1 = np.dot(np.dot(U1, d0), np.conjugate(U1.T))
d0_exp2 = np.dot(np.dot(U2, d0), np.conjugate(U2.T))
#d0_exp2 = d0_exp2/np.trace(d0_exp2)

# Calculamos la solucion
tiempo = np.linspace(0, 100, 1000)