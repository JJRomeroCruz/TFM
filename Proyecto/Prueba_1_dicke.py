#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:28:54 2024

@author: juanjo
"""
import numpy as np
import matplotlib.pyplot as plt
import qutip as q
import dicke
import time
import general


inicio = time.time()

# Parámetros del sistema
N = 2  # Número de átomos
omega_a = 0.1  # Frecuencia de transición de los átomos
omega = 1.0*omega_a  # Frecuencia del modo del campo electromagnético
#g = 1.0  # Constante de acoplamiento átomo-modo
g = 2*np.sqrt(omega_a)
kappa = 1*omega_a
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
psi0 = ini

#d0 = Qobj(d0)
# Lista de tiempos de la simulación
tlist = np.linspace(0, 5, 100)

# Sacamos el lindbladiano y sus autovalores
L = q.liouvillian(H, [J])
todoh = (L.dag()).eigenstates(sparse = True, sort = 'high', eigvals = 4)

vals = todoh[0]
vects = todoh[1]

# Efecto Mpemba
U1 = general.Mpemba1_mejorada_q(L, vects, vals, d0, N, ini)[0]
U2 = general.Mpemba2_mejorada_q(L, vects, vals, d0, N, ini)[0]

l1 = np.reshape(vects[1], (d0.shape[0], d0.shape[1]))
posibles, traza = general.buscar_angulos(l1, d0.full(), N)
theta, phi = posibles[traza.index(min(traza))]
U3 = general.Mpemba_sep(theta, phi, N)
d03 = np.dot(np.dot(U3, d0), np.conjugate(U3.T))

#print("dimension: ", q.Qobj(np.dot(U2.full(), ini.full())))
#print("dimension: ", U2.shape, ini.shape)

# Resolver la ecuación maestra para obtener la evolución temporal del sistema
#output = mesolve(H, psi0, tlist, collapse_operators, [a.dag() * a, sigma_minus.dag() * sigma_minus])
print(d0.shape)
ini1 = np.dot(U1.full(), ini.full())
#ini2 = ini
ini2 = np.dot(U2.full(), ini.full())
d01 = np.dot(np.dot(U1.full(), d0.full()), (U1.dag()).full())
#d02 = d01
d02 = np.dot(np.dot(U2.full(), d0.full()), (U2.dag()).full())
print(ini1.shape == ini.shape)
print(H.shape)
output1 = q.mesolve(H, q.Qobj(ini), tlist, [J])
output2 = q.mesolve(q.Qobj(H.full()), q.Qobj(d01), tlist, [q.Qobj(J.full())])
output3 = q.mesolve(q.Qobj(H.full()), q.Qobj(d02), tlist, [q.Qobj(J.full())])
output4 = q.mesolve(q.Qobj(H.full()), q.Qobj(d03), tlist, [q.Qobj(J.full())])

est = q.steadystate(q.Qobj(H), [q.Qobj(J)])
vector1, vector2, vector3, vector4 = [], [], [], []
for i in range(len(output1.states)):
    x1 = output1.states[i] - est
    x2 = output2.states[i] - q.Qobj(est.full())
    x3 = output3.states[i] - q.Qobj(est.full())
    x4 = output4.states[i] - q.Qobj(est.full())
    vector1.append(np.sqrt(np.trace(x1.dag()*x1)))
    vector2.append(np.sqrt(np.trace(x2.dag()*x2)))
    vector3.append(np.sqrt(np.trace(x3.dag()*x3)))
    vector4.append(np.sqrt(np.trace(x4.dag()*x4)))

"""
for state in output.states:
    x = state - est
    vector.append(np.sqrt(np.trace(x.dag()*x)))
"""
"""
# Graficar la ocupación del modo del campo electromagnético en función del tiempo
plt.plot(tlist, output.expect[0], label='Modo del campo electromagnético')
plt.plot(tlist, output.expect[1], label='Átomos en estado excitado')
plt.xlabel('Tiempo')
plt.ylabel('Ocupación')
plt.legend()
plt.show()
"""
fin = time.time()
print('Tiempo: ' + str(fin-inicio))
plt.plot(tlist, vector1, 'bo', label = 'Normal')
plt.plot(tlist, vector2, 'ro', label = 'Mpemba 1')
plt.plot(tlist, vector3, 'go', label = 'Mpemba 2')
plt.plot(tlist, vector4, 'yo', label = 'Mpemba_ang')
plt.legend()
plt.title('g = ' + str(g))
plt.show()