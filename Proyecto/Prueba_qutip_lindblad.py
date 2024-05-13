#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:11:04 2024

@author: juanjo
"""
import time
import numpy as np
from qutip import tensor, sigmax, sigmay, destroy, identity, spre, spost, liouvillian, Qobj
import qutip as q
import general
import dicke
import matplotlib.pyplot as plt

def dicke_lindbladiano(N, gamma, omega):
    # Definir los operadores
    sx = sigmax()
    sy = sigmay()
    sm = destroy(N)
    id_N = identity(N)
    
    # Construir los operadores del sistema
    Jx = tensor(sx, id_N)
    Jy = tensor(sy, id_N)
    #4Jz = tensor(0.5 * (sm.dag() * sm - id_N), id_N)
    a = tensor(sm, id_N)
    #print(a.shape, Jx.shape, Jy.shape)
    a = sm
    print(a.shape, Jx.shape, Jy.shape)

    # Construir el Hamiltoniano del modelo de Dicke
    H = omega * (Jx + a.dag() + a)
    
    # Construir los términos de Lindblad
    collapse_operators = [np.sqrt(gamma) * a]
    
    # Construir el superoperador de Lindblad
    Lindbladiano = liouvillian(H, collapse_operators)
    
    return Lindbladiano

# Parámetros del modelo de Dicke
inicio = time.time()


# Generamos el lindbladiano con el modelo de dicke
N = 2
#sigma = 1.0
#k = 0.1*sigma
#g = 1.0*sigma
#w = 0.01*sigma
sigma = 1.0
k = 0.5*sigma*N
g = 0.1*sigma*N
w = 0.1*sigma*np.sqrt(N)
params = [sigma, w, k, g]

H, J = dicke.dicke(N, params)

#Lindbladiano
L, b = general.Limblad(H, [J])

# Pasamos el lindbladiano a matriz de numpy
L = np.matrix(L, dtype = complex)

# Pasamos el lindbladiano a quantum object
Lq = q.Qobj(L)

print(type(Lq))
vects = Lq.eigenstates(sparse = False, sort = 'high', eigvals=2)

# Sacamos el estado estacionario
print((vects[1][0]).shape)
dim = int(np.sqrt(max((vects[1][0]).shape)))
r = np.reshape(vects[1][0], (dim, dim))
print(np.trace(np.dot(np.conjugate(r.T), r)))
r1 = q.steadystate(q.Qobj(H), [q.Qobj(J)])
r1 = np.reshape(r1, (dim, dim))
print(np.allclose(r, r1))
print('r1: ', r1.shape)
#print('r: ', r)
fin = time.time()
#print([Lq.overlap(vects[1][i]) for i in range(Lq.shape[0])])

# Ahora, vamos a tratar de resolver la master equation con qutip
t = np.linspace(0, 100, 1000)
d0, ini = dicke.densidad(N)
res = q.mesolve(q.Qobj(H), q.Qobj(d0), t, [q.Qobj(J)])

print(res.states)
 
estado = [elemento - r1 for elemento in res.states]
plt.plot(t, [np.sqrt(np.trace(np.dot(elemento.dag(), elemento))) for elemento in estado], label = 'ground')
print('Tiempo: ' + str(fin-inicio))
