#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:44:07 2024

@author: juanjo
"""
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parámetros del sistema
N = 40  # Número de átomos
omega = 1.0  # Frecuencia del modo del campo electromagnético
omega_a = 1.0  # Frecuencia de transición de los átomos
g = 0.1  # Constante de acoplamiento átomo-modo
kappa = 0.1  # Tasa de pérdida de fotones del modo del campo electromagnético

# Operadores de los átomos y del modo del campo electromagnético
a = tensor(destroy(N), qeye(2))
sigma_minus = tensor(qeye(N), destroy(2))

# Hamiltoniano del sistema
H0 = omega * a.dag() * a + omega_a * sigma_minus.dag() * sigma_minus
Hint = g * (a.dag() + a) * (sigma_minus + sigma_minus.dag())
H = H0 + Hint

# Operador de colapso para la pérdida de fotones del modo del campo electromagnético
collapse_operators = [np.sqrt(kappa) * a]

# Estado inicial: todos los átomos en el estado excitado y el modo del campo electromagnético en el estado de vacío
psi0 = tensor(basis(N, 0), basis(2, 1))

print(H.shape, a.shape, psi0.shape)

# Lista de tiempos de la simulación
tlist = np.linspace(0, 10, 100)

# Resolver la ecuación maestra para obtener la evolución temporal del sistema
output = mesolve(H, psi0, tlist, collapse_operators, [a.dag() * a, sigma_minus.dag() * sigma_minus])

# Graficar la ocupación del modo del campo electromagnético en función del tiempo
plt.plot(tlist, output.expect[0], label='Modo del campo electromagnético')
plt.plot(tlist, output.expect[1], label='Átomos en estado excitado')
plt.xlabel('Tiempo')
plt.ylabel('Ocupación')
plt.legend()
plt.show()
