#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 12:09:52 2023

@author: juanjo
"""
""" En este programa simplemente se va a sacar la matriz densidad para un numero
de spines N dado"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import scipy as sc
import random

# Funcion que me hace el producto de kronecker de las matrices identidad y ponemos en la posicion pos una matriz dada
def kronecker(matriz, pos, n):
  id = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype = complex)
  #sx = np.matrix([[0.0, 1.0], [1.0, 0.0]], dtype = complex)
  #sy = np.matrix([[0.0, -np.complex(0, 1.0)], [np.complex(0, 1.0), 0.0]], dtype = complex)
  #sz = np.matrix([[1.0, 0.0], [0.0, -1.0]], dtype = complex)
  res = id
  i = 0
  for i in range(n):
    if(i == pos):
      res = np.kron(res, matriz)
    else:
      res = np.kron(res, matriz)
  return res

# Funcion que me hace el ketbra de dos vectores
def ketbra(v1, v2):
  v2_conj = np.conjugate(v2)
  m = np.zeros((max(v1.shape[0], v1.shape[1]), max(v2.shape[0], v2.shape[1])), dtype = complex)
  for i in range(len(v1)):
    for k in range(len(v2)):
      m[i, k] = v1[i]*v2_conj[k]
  return m

# Funcion que me genera una matriz densidad inicial a partir de una combinacion lineal de los autoestados de Sz
def densidad(N):
  sz = np.matrix([[1.0, 0.0], [0.0, -1.0]], dtype = complex)
  id = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype = complex)

  # Construimos el operador sz
  suma = kronecker(0.5*sz, 0, N)
  i = 1
  while(i < N):
    suma += kronecker(0.5*sz, i, N)
    i += 1
  spin = suma
  #print(spin)

  # Creamos la base de autoestados de sz
  base = [np.linalg.eig(spin)[1][:, i] for i in range(spin.shape[0])]
  #print(base)

  # Ya estan normalizados
  # Construimos el vector inicial
  a, b = random.random(), random.random()
  #a = random.random()
  #b = np.sqrt(1.0-a*a)
  ini = (a + b*1.j)*base[0]
  for i in range(1, len(base)):
    a, b = random.random(), random.random()
    #a = random.random()
    #b = np.sqrt(1.0-a*a)
    ini += (a + b*1.j)*base[i]
    print(i)
  print(type(ini))
  print(ini.shape)
  # Le hacemos el ketbra para construir la matriz densidad
  d = ketbra(np.array(ini), np.array(ini))

  return d/np.trace(d)

N = 10 # Numero de spines (Para 20 no se puede)

d0 = densidad(N) # Matriz densidad inicial

""" Una vez calculada la matriz densidad, la pasamos a un fichero """

# Creamos en fichero
f = open('densidad.dat', 'x')
f.close()

def guardar_matriz_en_archivo(matriz, nombre_archivo):
    try:
        with open(nombre_archivo, 'w') as archivo:
            for fila in matriz:
                fila_str = ' '.join(map(str, fila))
                archivo.write(fila_str + '\n')
        print(f"Matriz guardada exitosamente en {nombre_archivo}")
    except Exception as e:
        print(f"Error al guardar la matriz en el archivo: {e}")

# Ejemplo de uso:
matriz_entrada = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

nombre_archivo_salida = 'densidad.dat'
guardar_matriz_en_archivo(d0, nombre_archivo_salida)
