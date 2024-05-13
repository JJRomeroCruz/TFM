#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:09:49 2024

@author: juanjo
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, destroy, sigmax, sigmaz, mesolve
import dicke
import qutip as q

# Parámetros del sistema
N = 2  # Número de espines (qubits)
omega = 1.0  # Frecuencia del campo electromagnético
Omega = 0.5  # Frecuencia de acoplamiento espín-campo
gamma = 0.1  # Tasa de decaimiento del campo electromagnético

# Definir los operadores
a = destroy(N)  # Operador de aniquilación del campo electromagnético
sz = tensor(sigmaz(), q.qeye(2))  # Operador de Pauli Z para el espín
sx = tensor(sigmax(), q.qeye(2))  # Operador de Pauli X para el espín

# Definir el hamiltoniano
H = Omega * sz + omega * (a.dag() * a) + 1/np.sqrt(N) * (a + a.dag()) * sx

# Definir el operador de colapso
c_ops = [np.sqrt(gamma) * a]

# Estado inicial (qubits en estado base y campo en estado coherente)
#psi0 = tensor(basis(2, 0), basis(2, 0), (basis(N, 0) + basis(N, 1)).unit())
d0, ini = dicke.densidad(N)

# Tiempos en los que se evaluará la evolución temporal
tiempos = np.linspace(0, 100, 1000)

# Resolver la ecuación maestra de Lindblad
#resultados = mesolve(H, q.Qobj(d0), tiempos, c_ops, [sx, sz])
resultados = mesolve(H, q.Qobj(d0), tiempos, c_ops)
r1 = q.steadystate(H, c_ops)

estado = [elemento - r1 for elemento in resultados.states]
plt.plot(tiempos, [np.sqrt(np.trace(np.dot(elemento.dag(), elemento))) for elemento in estado], label = 'ground')
"""
# Graficar las expectativas de los observables
plt.plot(resultados.times, resultados.expect[0], label='Sx')
plt.plot(resultados.times, resultados.expect[1], label='Sz')
plt.xlabel('Tiempo')
plt.ylabel('Expectativa')
plt.title('Evolución temporal de observables en el modelo de Dicke')
plt.legend()
plt.show()
"""