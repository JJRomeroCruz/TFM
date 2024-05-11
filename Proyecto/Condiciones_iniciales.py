#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:00:56 2024

@author: juanjo
"""
import numpy as np
import random as random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# Funcion que me hace el ketbra de dos vectores
def ketbra(v1, v2):
  v2_conj = np.conjugate(v2)
  m = np.zeros((max(v1.shape[0], v1.shape[1]), max(v2.shape[0], v2.shape[1])), dtype = complex)
  for i in range(len(v1)):
    for k in range(len(v2)):
      m[i, k] = v1[i]*v2_conj[k]
  return m

nveces = 100000
lista1 = []
lista2 = []
p = []
N = 10
for i in range(nveces):
    #a1, b1 = random.random(), random.random()
    #a2, b2 = random.random(), random.random()
    v = []
    for i in range(N):
        a1, b1 = random.random(), random.random()
        v.append(a1 + 1.j*b1)
    v = np.asarray(v, dtype = complex)
    v = np.reshape(v, (1, len(v)))
    d = ketbra(v.T, v.T)
    #a1, b1 = -1.0 + 2.0*random.random(), -1.0 + 2.0*random.random()
    #a2, b2 = -1.0 + 2.0*random.random(), -1.0 + 2.0*random.random()
    
    #a1, b1 = -10.0 + 20.0*random.random(), -10.0 + 20.0*random.random()
    #a2, b2 = -10.0 + 20.0*random.random(), -10.0 + 20.0*random.random()
    
    #lista1.append(a)
    #lista2.append(b)
    #p.append(np.angle(a1*a2 + b1*b2 + 1.j*(b1*a2-b2*a1)))
    #p.append(np.sqrt((a1*a2 + b1*b2)**2 + (b1*a2 - b2*a1)**2))
    p.append(np.trace(d))
    

# Crear el histograma
plt.hist(p, bins=30, alpha=0.75, color='blue', edgecolor='black')

# Añadir títulos y etiquetas
plt.title('Histograma de Frecuencias')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')

# Mostrar el histograma
plt.show()


    