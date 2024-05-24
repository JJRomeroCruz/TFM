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
from mpl_toolkits.mplot3d import Axes3D


inicio = time.time()

# Parámetros del sistema
N = 40 # Número de átomos
omega_a = 1 # Frecuencia de transición de los átomos
omega = 1.0*omega_a  # Frecuencia del modo del campo electromagnético
g = 1.0*omega_a  # Constante de acoplamiento átomo-modo
#g = np.sqrt(16)*np.sqrt(omega_a)
kappa = 5*omega_a
params = [omega_a, omega, kappa, g]
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
print(H.shape)
#d0 = Qobj(d0)
# Lista de tiempos de la simulación
tlist = np.linspace(0, 5, 100)

# Sacamos el lindbladiano y sus autovalores
L = q.liouvillian(H, [J])
todoh = (L.dag()).eigenstates(sparse = True, sort = 'high', eigvals = N+3)

vals = todoh[0]
vects = todoh[1]

# Efecto Mpemba
U1 = general.Mpemba1_mejorada_q(L, vects, vals, d0, N, ini)[0]
U2, U_cambio, vals_l = general.Mpemba2_mejorada_q(L, vects, vals, d0, N, ini)
"""
l1 = np.reshape(vects[1], (d0.shape[0], d0.shape[1]))
posibles, traza = general.buscar_angulos(l1, d0.full(), N)
theta, phi = posibles[traza.index(min(traza))]
U3 = general.Mpemba_sep(theta, phi, N)
d03 = np.dot(np.dot(U3.full(), d0), (U3.dag()).full())
"""
#d03 = U3*d0*U3.dag()
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
#output4 = q.mesolve(q.Qobj(H.full()), q.Qobj(d03), tlist, [q.Qobj(J.full())])

est = q.steadystate(q.Qobj(H), [q.Qobj(J)])
vector1, vector2, vector3, vector4 = [], [], [], []
for i in range(len(output1.states)):
    x1 = output1.states[i] - est
    x2 = output2.states[i] - q.Qobj(est.full())
    x3 = output3.states[i] - q.Qobj(est.full())
#    x4 = output4.states[i] - q.Qobj(est.full())
    vector1.append(np.sqrt(np.trace(x1.dag()*x1)))
    vector2.append(np.sqrt(np.trace(x2.dag()*x2)))
    vector3.append(np.sqrt(np.trace(x3.dag()*x3)))
#    vector4.append(np.sqrt(np.trace(x4.dag()*x4)))

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
"""
plt.plot(tlist, vector1, 'b.-', label = 'Normal')
plt.plot(tlist, vector2, 'r.-', label = 'Mpemba 1')
plt.plot(tlist, vector3, 'go', label = 'Mpemba 2')
plt.plot(tlist, vector4, 'y.-', label = 'Mpemba_ang')
plt.legend()
plt.title('g = ' + str(g))
plt.show()

# Generamos la grafica de las posibles transformaciones individuales
radio = 1.0
theta_rango = (0, np.pi)
phi_rango = (0, 2*np.pi)


posibles_filtro, traza_filtro = [], []
for i in range(len(traza)):
    if(traza[i] < 5e-2):
        traza_filtro.append(traza[i])
        posibles_filtro.append(posibles[i])

theta_values = [tup[0] for tup in posibles_filtro]
phi_values = [tup[1] for tup in posibles_filtro]

x1, y1, z1 = general.parametrizacion_esfera(radio, theta_rango, phi_rango)
x2, y2, z2 = general.esfera_partes(radio, theta_values, phi_values)
"""
fig = plt.figure(figsize = (12, 12))

# Parte distancias
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(tlist, vector1, 'b.-', label = 'Normal')
#ax1.plot(tlist, vector2, 'r.-', label = 'Mpemba 1')
ax1.plot(tlist, vector3, 'g.-', label = 'Mpemba 2')
#ax1.plot(tlist, vector4, 'y.-', label = 'Mpemba_ang')

ax1.grid(True)
ax1.legend()
#ax1.set_title('Distancia de H-S. Dicke. g = ' + str(g))


# Parte esfera
ax2 = fig.add_subplot(3, 1, 2)

s = np.linspace(0, np.pi*0.5, 1000)
c_s = vals_l[0]*np.cos(s)**2 + vals_l[1]*np.sin(s)**2
ax2.plot(s, c_s)
ax2.plot(s, np.zeros_like(c_s))

ax2.set_xlabel('s')
ax2.set_ylabel('c(s)')

"""
# Parte valor esperado del hamiltoniano respecto del tiempo
ax3 = fig.add_subplot(3, 1, 3)
expect_H1 = q.expect(q.Qobj(H), output1.states)
expect_H2 = q.expect(q.Qobj(H.full()), [q.Qobj(elemento.full()) for elemento in output2.states])
expect_H3 = q.expect(q.Qobj(H.full()), [q.Qobj(elemento.full()) for elemento in output3.states])

#print(expect_H1)

ax3.plot(tlist, expect_H1, label = 'Normal')
ax3.plot(tlist, expect_H2, label = 'Mpemba1')
#ax3.plot(tlist, expect_H3, label = 'Mpemba2')
ax3.legend()
"""
plt.show()