#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:49:55 2024

@author: juanjo
"""
import numpy as np
import sympy as sp
import scipy as sc
import random as random
import matplotlib.pyplot as plt
from general import ketbra, kronecker, matriz_cuadrada, H_eff, FL_vector, Limblad, Calcularp, NuevoEstado, segunda_posicion_mas_grande, Mpemba1, Mpemba2
from general import indices_mayores_que_100, estacionario, ResolverSistema, eliminar_duplicados, generar_base_ortonormal, Mpemba2_mejorada, Mpemba1_mejorada
from general import estacionario_bueno, solucion, Mpemba_sep, Mpemba1_mejorada, Mpemba2_mejorada, buscar_segundo_maximo

""" En este script se definen las funciones que tienen que ver con el modelo de qubits en el que estan todos fuertemente correlacionados """

# Funcion que me calcula el hamiltoniano y el operador de salto del modelo de qubits Rydeberg
def rydberg(N, params):
    iden = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype = complex)
    sx = 0.5*np.matrix([[0.0, 1.0], [1.0, 0.0]], dtype = complex)
    sy = 0.5*np.matrix([[0.0, -1.j], [1.j, 0.0]], dtype = complex)
    sz = 0.5*np.matrix([[1.0, 0.0], [0.0, -1.0]], dtype = complex)
    
    sigma = params[0]
    delta = params[1]
    v = params[2]
    k = params[3]
    
    suma1 = kronecker(sx, 0, N)
    suma2 = kronecker(sz, 0, N)
    suma3 = kronecker(np.dot(sz, sz), 0, N)
    suma4 = kronecker(sx - 1.j*sy, 0, N)
    
    i = 1
    while(i < N):
        suma1 +=  kronecker(sx, i, N)
        suma2 += kronecker(sz, i, N)
        suma3 += kronecker(np.dot(sz, sz), i, N)
        suma4 += kronecker(sx - 1.j*sy, i, N)
        i += 1 
        
    H = sigma*suma1 - delta*suma2 + (v/(1.0*N))*suma3
    J = np.sqrt(k/N)*suma4
    
    return H, J
# Funcion que me saca la matriz densidad inicial del modelo de átomos de Rydberg
def densidad_rydberg(N, H):
  sz = np.matrix([[1.0, 0.0], [0.0, -1.0]], dtype = complex)
  iden = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype = complex)

  # Construimos el operador sz
  suma = kronecker(0.25*np.dot(sz, sz), 0, N)
  i = 1
  while(i < N):
    suma += kronecker(0.25*np.dot(sz, sz), i, N)
    i += 1
  spin = suma
  #print(spin)

  # Creamos la base de autoestados de sz
  todo = sp.Matrix(H, dtype = complex).eigenvects()
  #todo = sp.Matrix(spin, dtype = complex).eigenvects()
  base = [np.array(tup[2], dtype = complex) for tup in todo]
  #base = [np.linalg.eig(H)[1][:, i] for i in range(spin.shape[0])]
  #print(base)

  # Construimos el vector inicial
  a, b = random.random(), random.random()
  ini = (a + b*1.j)*base[0]
  for i in range(1, len(base)):
    a, b = random.random(), random.random()
    ini += (a + b*1.j)*base[i]
  # Le hacemos el ketbra para construir la matriz densidad

  ini = ini/(np.linalg.norm(ini))
  d = ketbra(ini, ini)
  res = d/(np.linalg.norm(d))
  return res, ini

