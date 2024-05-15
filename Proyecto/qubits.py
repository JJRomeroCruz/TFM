#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:55:15 2024

@author: juanjo
"""
import numpy as np
import sympy as sp
import scipy as sc
import matplotlib.pyplot as plt
import qutip as q
import dicke
import general
import random as random

# Funcion que me crea el hamiltoniano del modelo de los qubits
def qubits(N, params):
  id = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype = complex)
  sx = np.matrix([[0.0, 1.0], [1.0, 0.0]], dtype = complex)
  sy = np.matrix([[0.0, -1.j], [1.j, 0.0]], dtype = complex)
  sz = np.matrix([[1.0, 0.0], [0.0, -1.0]], dtype = complex)

  sigma = params[0]
  k = params[1]

  suma1 = general.kronecker(0.5*sx, 0, N)
  suma2 = general.kronecker(0.5*(sx - 1.j*sy), 0, N)

  i = 1
  while(i < N):
    #print(i)
    suma1 += general.kronecker(0.5*sz, i, N)
    suma2 += general.kronecker(0.5*(sx - 1.j*sy), i, N)
    i += 1

  H = sigma*suma1
  J = np.sqrt(k)*suma2

  return H, J

def densidad(N):
  sx = np.matrix([[0.0, -1.0], [1.0, 0.0]], dtype = complex)
  id = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype = complex)

  # Construimos el operador sz
  suma = general.kronecker(0.5*sx, 0, N)
  i = 1
  while(i < N):
    suma += general.kronecker(0.5*sx, i, N)
    i += 1
  spin = suma

  # Creamos la base de autoestados de sz
  base = [np.linalg.eig(spin)[1][:, i] for i in range(spin.shape[0])]

  # Construimos el vector inicial
  a, b = random.random(), random.random()
  ini = (a + b*1.j)*base[0]
  for i in range(1, len(base)):
    a, b = random.random(), random.random()
    ini += (a + b*1.j)*base[i]
  # Le hacemos el ketbra para construir la matriz densidad

  ini = ini/(np.linalg.norm(ini))
  d = general.ketbra(np.array(ini), np.array(ini))
  res = d/(np.linalg.norm(d))
  return res, ini
