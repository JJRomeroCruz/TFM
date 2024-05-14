#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:40:25 2024

@author: juanjo
"""
""" En este script se van a sacar todos los posibles ángulos ploteados en la esfera para el modelo de Dicke """
import numpy as np
import random as random
import matplotlib.pyplot as plt
import qutip as q 
import dicke
import general
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
import time

inicio = time.time()

# Generamos el hamiltoniano y los operadores de salto
N = 2
sigma = 1.0
w = 1.0*sigma
k = 1.0*sigma
#g = 5.0*sigma
g = np.sqrt(2)*sigma
params = [sigma, w, k, g]
H, J = dicke.dicke(N, params)

# Matriz densidad inicial y vector inicial
d0, ini = dicke.densidad(N)
#ini = np.zeros(int(2**N))
#ini[0] = 1
#d0 = general.ketbra(ini, ini)

# Sacamos el lindbladiano y lo diagonalizamos
"""
L, b = general.Limblad(H, [J])
todo = L.eigenvects()
todoh = (L.H).eigenvects()
vals = np.array([tup[0] for tup in todo], dtype = complex)
"""

# Sacamos el lindbladiano y lo diagonalizamos
L, b = general.Limblad(H, [J])
L = np.matrix(L, dtype = complex)
# Diagonalizamos
Lq = q.Qobj(L)
Lqh = q.Qobj(L.H)
todo = Lq.eigenstates(sparse = False, sort = 'high')
todoh = Lqh.eigenstates(sparse = False, sort = 'high')
vals = todo[0]

"""
# autoMatrices derecha
r = [np.asarray(tup[2], dtype = complex) for tup in todo]
r = [np.squeeze(np.asarray(elemento, dtype = complex)) for elemento in r]
r = [elemento/np.linalg.norm(elemento) for elemento in r]
r = [np.reshape(elemento, (d0.shape[0], d0.shape[1])) for elemento in r]

# automatrices izquierda
l = [np.asarray(tup[2], dtype = complex) for tup in todoh]
l = [np.squeeze(np.asarray(elemento, dtype = complex)) for elemento in l]
l = [elemento/np.linalg.norm(elemento) for elemento in l]
l = [np.reshape(elemento, (d0.shape[0], d0.shape[1])) for elemento in l]
"""

r = [np.reshape(elemento, d0.shape) for elemento in todo[1]]
l = [np.reshape(elemento, d0.shape) for elemento in todoh[1]]

# Sacamos todos los posibles angulos
segundo_maximo, indice_segundo_maximo = general.buscar_segundo_maximo(list(np.real(vals)))
L1 = l[indice_segundo_maximo]
posibles, traza = general.buscar_angulos(L1, d0, N)
posibles_filtro, traza_filtro = [], []
for i in range(len(traza)):
    if(traza[i] < 1e-3):
        traza_filtro.append(traza[i])
        posibles_filtro.append(posibles[i])


# Los representamos
radio = 1.0
theta_rango1 = (0, np.pi)  # Rango de ángulo polar de 0 a pi
phi_rango1 = (0, 2 * np.pi)  # Rango de ángulo azimutal de 0 a 2*pi

theta_values = [tup[0] for tup in posibles_filtro]
phi_values = [tup[1] for tup in posibles_filtro]


# Obtener las coordenadas para la parametrización
x1, y1, z1 = general.parametrizacion_esfera(radio, theta_rango1, phi_rango1)
x2, y2, z2 = general.esfera_partes(radio, theta_values, phi_values)
fin = time.time()
print('Tiempo: ' + str(fin-inicio))

# Graficar la esfera
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, y1, z1, color='b', alpha= 0.3)
#ax.plot_surface(x2, y2, z2, cmap = 'viridis', color = 'b', alpha = 0.9) # Alpha es la opacidad
#ax.plot_surface(x2, y2, z2, color = 'r', alpha = 1.0) # Alpha es la opacidad
ax.scatter(x2, y2, z2, c='r', marker='o', label='Puntos en la esfera')
# Cambiar la orientación de la vista
ax.view_init(elev=45, azim=45)  # Elevación y azimut en grados
# Configuración adicional para mejorar la visualización
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Posibles transformaciones. Modelo de Dicke. N = ' + str(N) + ', k = ' + str(k) + ' g = ' +  str(g))

plt.show()


