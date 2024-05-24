#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:44:07 2024

@author: juanjo
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qutip import *

# Definición de parámetros
N = 2  # Número de átomos
omega = 1.0  # Frecuencia del modo del campo
omega0 = 1.0  # Frecuencia de transición atómica
lambda_c = 1.0  # Fuerza de acoplamiento

# Definición de operadores
a = tensor(destroy(N), qeye(2))
adag = tensor(create(N), qeye(2))
Jz = tensor(qeye(N), sigmaz()/2)
Jp = tensor(qeye(N), sigmap())
Jm = tensor(qeye(N), sigmam())

# Hamiltoniano del modelo de Dicke
H = omega * adag * a + omega0 * Jz + (lambda_c / np.sqrt(N)) * (adag + a) * (Jp + Jm)

# Rango de valores para las posiciones del campo
x_values = np.linspace(-10, 10, 100)
y_values = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x_values, y_values)
Z = np.zeros_like(X)

# Cálculo del paisaje energético
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        alpha = X[i, j] + 1j * Y[i, j]
        coherent_state = tensor(coherent(N, alpha), basis(2, 0))
        Z[i, j] = (coherent_state.dag() * H * coherent_state).full().real

# Visualización del paisaje energético
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('Re(α)')
ax.set_ylabel('Im(α)')
ax.set_zlabel('Energía')
plt.title('Energy Landscape del Modelo de Dicke')
plt.show()
