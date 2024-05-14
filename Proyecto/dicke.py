#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:45:59 2024

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

# Funcion que me crea el hamiltoninano del modelo de Dicke dado un cierto numero de spines y su operador salto
def dicke(N, params):
  id = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype = complex)
  sx = np.matrix([[0.0, 1.0], [1.0, 0.0]], dtype = complex)
  sy = np.matrix([[0.0, -1.j], [1.j, 0.0]], dtype = complex)
  sz = np.matrix([[1.0, 0.0], [0.0, -1.0]], dtype = complex)

  sigma = params[0]
  w = params[1]
  k = params[2]
  g = params[3]

  suma1 = kronecker(0.5*sz, 0, N)
  #suma2 = kronecker(0.25*np.dot(sx, sx), 0, N)
  suma2 = kronecker(0.5*sx, 0, N)

  i = 1
  while(i < N):
    #print(i)
    suma1 += kronecker(0.5*sz, i, N)
    #suma2 += kronecker(0.25*np.dot(sx, sx), i, N)
    suma2 += kronecker(0.5*sx, i, N)
    i += 1

  H = sigma*suma1 - ((4.0*w*g*g)/(4.0*w**2 + k**2))*(1.0/N)*np.dot(suma2, suma2)
  J = ((2.0*np.abs(g)*np.sqrt(k))/(np.sqrt(N*(4*w*w + k*k))))*suma2

  return H, J


# Funcion que me genera una matriz densidad inicial a partir de una combinacion lineal de los autoestados de Sz
def densidad(N):
  sz = np.matrix([[1.0, 0.0], [0.0, -1.0]], dtype = complex)
  id = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype = complex)
  sx = np.matrix([[0.0, 1.0], [1.0, 0.0]], dtype = complex)
  sy = np.matrix([[0.0, -1.j], [1.j, 0.0]], dtype = complex)
  
  # Construimos el operador sz
  suma = kronecker(0.5*sz, 0, N)
  i = 1
  while(i < N):
    suma += kronecker(0.5*sz, i, N)
    i += 1
  spin = suma
  #print(spin)

  # Creamos la base de autoestados de sz
  todo = sp.Matrix(spin, dtype = complex).eigenvects()
  base = [np.array(tup[2], dtype = complex) for tup in todo]
  #base = [np.linalg.eig(spin)[1][:, i] for i in range(spin.shape[0])]
  #print(base)

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

# Funcion que me genera una matriz densidad inicial a partir de una combinacion lineal de los autoestados de Sz
def densidad2(N):
  sz = np.matrix([[1.0, 0.0], [0.0, -1.0]], dtype = complex)
  id = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype = complex)
  sx = np.matrix([[0.0, 1.0], [1.0, 0.0]], dtype = complex)
  sy = np.matrix([[0.0, -1.j], [1.j, 0.0]], dtype = complex)
  
  # Construimos el operador sz
  suma = kronecker(0.25*np.dot(sz, sz) + 0.25*np.dot(sy, sy) + 0.25*np.dot(sx, sx), 0, N)
  i = 1
  while(i < N):
    suma += kronecker(0.25*np.dot(sz, sz) + 0.25*np.dot(sy, sy) + 0.25*np.dot(sx, sx), i, N)
    i += 1
  spin = suma
  #print(spin)

  # Creamos la base de autoestados de sz
  todo = sp.Matrix(spin, dtype = complex).eigenvects()
  base = [np.array(tup[2], dtype = complex) for tup in todo]
  #base = [np.linalg.eig(spin)[1][:, i] for i in range(spin.shape[0])]
  #print(base)

  # Construimos el vector inicial
  a, b = random.random(), random.random()
  ini = (a + b*1.j)*base[0]
  for i in range(1, len(base)):
    a, b = random.random(), random.random()
    ini += (a + b*1.j)*base[i]
  # Le hacemos el ketbra para construir la matriz densidad

  ini = ini/(np.linalg.norm(ini))
  d = ketbra(np.array(ini), np.array(ini))
  #res = d/(np.linalg.norm(d))
  return d, ini

