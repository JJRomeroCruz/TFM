#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 00:26:16 2024

@author: juanjo
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import qutip as q
import ising
import dicke
import general
import time

"""
Aquí sacamos la figura 2 del artículo de rotaciones individuales
"""
inicio = time.time()

vector_v, vector_om, vector_l = [], [], []
a = 1
gam = 1
om = 0.1*gam
v = 0.1*gam
N = 4
om_inicial, om_final, om_salto = 0.1*gam, 4*gam, 0.2*gam
v_inicial, v_final, v_salto = 0.1*gam, 7*gam, 0.2*gam
N_om, N_v = round((om_final - om_inicial)/om_salto), round((v_final-v_inicial)/v_salto)
vector_l = np.zeros((N_om, N_v), dtype = float)

for i in range(N_om):
    #v = v_inicial
    for j in range(N_v):
        params = [om, v, a, gam]
        H, list_J = ising.ising(params, N)
        L = q.liouvillian(H, list_J)
        todo = L.eigenstates(sparse = True, sort = 'high', eigvals = 3)
        vals = todo[0]
        vector_l[i, j] = sorted(np.real(vals))[1]
        #print(vector_l[i, j])
        vector_v.append(v)
        vector_om.append(om)
        v += v_salto
    om += om_salto
    
y = np.linspace(om_inicial, om_final, N_om)
x = np.linspace(v_inicial, v_final, N_v)

#x, y = vector_om, vector_v

#for i in range(N_om):
#    for j in range()
    
"""
while(om < 4*gam):
    v = 0.1*gam
    while(v < 7*gam):
        params = [om, v, a, gam]
        H, list_J = ising.ising(params, N)
        L = q.liouvillian(H, list_J)
        vals = L.eigenstates(sparse = True, sort = 'high', eigvals = 3)[0]
        vector_l.append(vals[1])
        vector_v.append(v)
        vector_om.append(om)
        v += 0.5*gam
    om += 0.5*gam
"""
# Ahora, representamos
#x = vector_om
#y = vector_v
#z = np.real(vector_l)
z = vector_l
#x, y = np.meshgrid(x, y)
"""
fig, ax = plt.subplots()
im = ax.scatter(x, y, c = z, cmap = plt.cm.jet)
#ax = fig.add_subplot(111, projection = '3d')

#sc = ax.scatter(x, y, z, c = z, cmap = 'viridis', marker = 'o')
#ax.view_init(elev = 60.0, azim = 45)

fig.colorbar(im, ax = ax, label = r'Re($\lambda_1)$')

im.set_clim(0.0, 1.0)

ax.set_xlabel(r'$\omega/\gamma$ (Eje x)')
ax.set_ylabel(r'$V/\gamma$ (Eje y)')
ax.set_title(r'Parte real de $\lambda_1$')
"""

"""
OTRO INTENTO
fig, ax = plt.subplots()
im = ax.contourf(x, y, z, 100)
im2 = ax.contour(x, y, z, colors = 'k')
fig.colorbar(im, ax = ax)
plt.show()
"""
X, Y = np.meshgrid(x, y)

plt.figure(figsize = (8, 6))
cmap = plt.get_cmap('inferno')
heatmap = plt.pcolormesh(X, Y, z, shading = 'auto', cmap = 'plasma')
cbar = plt.colorbar(heatmap, label = r'Re($\lambda$)')
heatmap.set_clim(-1, 1)

"""
# Crear el mapa de calor
plt.figure()
plt.pcolormesh([x, y], z, shading='auto', cmap='viridis')
plt.colorbar(label='Valor de Z')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Mapa de calor')
"""

plt.show()

fin = time.time()
print('Tiempo: ' + str(fin-inicio))
