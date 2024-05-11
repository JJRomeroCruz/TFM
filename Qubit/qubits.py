#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:14:26 2024

@author: juanjo
"""

import numpy as np
import sympy as sp
import scipy as sc
import random as random
import matplotlib.pyplot as plt
from general.py import ketbra, kronecker, matriz_cuadrada, H_eff, FL_vector, Limblad, Calcularp, NuevoEstado, segunda_posicion_mas_grande, Mpemba1, Mpemba2
from general.py import indices_mayores_que_100, estacionario, ResolverSistema, eliminar_duplicados, generar_base_ortonormal, Mpemba2_mejorada, Mpemba1_mejorada


# Funcion que nos dice si la matriz l2 es hermitica segun los parametros que le pasemos
def L2hermitica(N, params, d):
  # Sacamos el hamiltoniano y los operadores de salto
  H, J = qubits(N, params)
  # Sacamos el lindbladiano
  L, b = Limblad(sp.Matrix(H, dtype = complex), [sp.Matrix(J, dtype = complex)])

  # Diagonalizamos el lindbladiano
  todo = (L.H).eigenvects()
  vals = np.array([tup[0] for tup in todo], dtype = complex)
  L_e = [tup[2] for tup in todo]

  # Normalizamos las automatrices
  L_e = [np.reshape(matriz, (d.shape[0], d.shape[1])) for matriz in L_e]

  # Extraemos la matriz por la izquierda cuyo autovalor es el mayor (y no es cero)
  lista_vals =  list(-np.real(vals))
  maximo = max(lista_vals)
  indice_maximo = lista_vals.index(maximo)
  lista_vals.remove(maximo)
  segundo_maximo = max(lista_vals)
  indice_segundo_maximo = lista_vals.index(segundo_maximo)
  if indice_segundo_maximo >= indice_maximo:
    indice_segundo_maximo += 1

  L1 = np.reshape(L_e[indice_segundo_maximo], (d.shape[0], d.shape[1]))
  L1 = sp.Matrix(L1, dtype = complex)
  # Una vez obtenida L2, vemos si es hermitica
  eshermitica = False
  if(L1 == L1.H):
    eshermitica = True
  return eshermitica, vals[indice_segundo_maximo]

# Funcion que me crea el hamiltoniano del modelo de los qubits
def qubits(N, params):
  id = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype = complex)
  sx = np.matrix([[0.0, 1.0], [1.0, 0.0]], dtype = complex)
  sy = np.matrix([[0.0, -1.j], [1.j, 0.0]], dtype = complex)
  sz = np.matrix([[1.0, 0.0], [0.0, -1.0]], dtype = complex)

  sigma = params[0]
  k = params[1]

  suma1 = kronecker(0.5*sx, 0, N)
  suma2 = kronecker(0.5*(sx - 1.j*sy), 0, N)

  i = 1
  while(i < N):
    #print(i)
    suma1 += kronecker(0.5*sz, i, N)
    suma2 += kronecker(0.5*(sx - 1.j*sy), i, N)
    i += 1

  H = sigma*suma1
  J = np.sqrt(k)*suma2

  return H, J

def densidad_qubit(N):
  sx = np.matrix([[0.0, -1.0], [1.0, 0.0]], dtype = complex)
  id = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype = complex)

  # Construimos el operador sz
  suma = kronecker(0.5*sx, 0, N)
  i = 1
  while(i < N):
    suma += kronecker(0.5*sx, i, N)
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
  d = ketbra(np.array(ini), np.array(ini))
  res = d/(np.linalg.norm(d))
  return res, ini