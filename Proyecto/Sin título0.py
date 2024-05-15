#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:28:54 2024

@author: juanjo
"""
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import dicke

# Parámetros del sistema
N = 5  # Número de átomos
omega = 1.0  # Frecuencia del modo del campo electromagnético
omega_a = 1.0  # Frecuencia de transición de los átomos
g = 1.0  # Constante de acoplamiento átomo-modo
kappa = 1
params = [omega, omega_a, kappa, g]
# Operadores de los átomos y del modo del campo electromagnético
#a = tensor(destroy(N), qeye(2))
#sigma_minus = tensor(qeye(N), destroy(2))

# Hamiltoniano del sistema
#H0 = omega * a.dag() * a + omega_a * sigma_minus.dag() * sigma_minus
#Hint = g * (a.dag() + a) * (sigma_minus + sigma_minus.dag())
#H = H0 + Hint
H, J = dicke.dicke_bueno(N, params)
# Operador de colapso para la pérdida de fotones del modo del campo electromagnético
#collapse_operators = [np.sqrt(kappa) * a]

# Estado inicial: todos los átomos en el estado excitado y el modo del campo electromagnético en el estado de vacío
#psi0 = tensor(basis(N, 0), basis(2, 1))
d0, ini = dicke.densidad_bueno(N)
#psi0 = Qobj(ini)
#d0 = Qobj(d0)
# Lista de tiempos de la simulación
tlist = np.linspace(0, 20, 100)
#print(psi0)
# Resolver la ecuación maestra para obtener la evolución temporal del sistema
#output = mesolve(H, psi0, tlist, collapse_operators, [a.dag() * a, sigma_minus.dag() * sigma_minus])
output = mesolve(Qobj(H), psi0, tlist, [Qobj(J)])

est = steadystate(Qobj(H), [Qobj(J)])
vector = []
for state in output.states:
    x = state - est
    vector.append(np.sqrt(np.trace(x.dag()*x)))
"""
# Graficar la ocupación del modo del campo electromagnético en función del tiempo
plt.plot(tlist, output.expect[0], label='Modo del campo electromagnético')
plt.plot(tlist, output.expect[1], label='Átomos en estado excitado')
plt.xlabel('Tiempo')
plt.ylabel('Ocupación')
plt.legend()
plt.show()
"""
plt.plot(tlist, vector, 'bo')