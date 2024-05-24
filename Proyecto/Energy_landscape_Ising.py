#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 19:05:41 2024

@author: juanjo
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qutip import *

# Definición de parámetros
N = 4  # Número de sitios
J = 1.0  # Constante de acoplamiento
h_values = np.linspace(0, 10, 100)  # Valores del campo magnético transversal
J_values = np.linspace(-5, 5, 100)  # Valores de la constante de acoplamiento

# Definición de operadores de Pauli
si = qeye(2)
sx = sigmax()
sz = sigmaz()

# Construcción del Hamiltoniano del modelo de Ising
def ising_hamiltonian(J, h, N):
    H = 0
    for i in range(N):
        H += -J * tensor([sz if j == i or j == (i+1)%N else si for j in range(N)])
    for i in range(N):
        H += -h * tensor([sx if j == i else si for j in range(N)])
    return H

# Matrices para almacenar las energías
energies = np.zeros((len(J_values), len(h_values)))

# Cálculo del paisaje energético
for i, J_val in enumerate(J_values):
    for j, h_val in enumerate(h_values):
        H = ising_hamiltonian(J_val, h_val, N)
        energies[i, j] = H.eigenenergies()[0]  # Energía del estado fundamental

# Visualización del paisaje energético
X, Y = np.meshgrid(J_values, h_values)
Z = energies.T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('J')
ax.set_ylabel('h')
ax.set_zlabel('Energía del estado fundamental')
plt.title('Paisaje Energético del Modelo de Ising')
plt.colorbar(surf)
plt.show()
