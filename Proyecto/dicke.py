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
import qutip as q
import general

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
  #J = ((2.0*np.abs(g)*np.sqrt(k))/(np.sqrt(N*(4*w*w + k*k))))*suma2
  J = -((np.sqrt(k)*g*(4*w + 2*1.j*k))/(np.sqrt(N)*(4*w*w + k*k)))*suma2
  return H, J

def dicke_q(N, params):
    iden = q.qeye(2)
    sx = q.sigmax()
    sy = q.sigmay()
    sz = q.sigmaz()
    
    sigma = params[0]
    w = params[1]
    k = params[2]
    g = params[3]

    suma1 = general.kronecker_q(0.5*sz, 0, N)
    suma2 = general.kronecker_q(0.5*sx, 0, N)

    i = 1
    while(i < N):
      print(i)
      suma1 += general.kronecker_q(0.5*sz, i, N)
      #suma2 += kronecker(0.25*np.dot(sx, sx), i, N)
      suma2 += general.kronecker_q(0.5*sx, i, N)
      i += 1

    H = sigma*suma1 - ((4.0*w*g*g)/(4.0*w**2 + k**2))*(1.0/N)*suma2*suma2
    #J = ((2.0*np.abs(g)*np.sqrt(k))/(np.sqrt(N*(4*w*w + k*k))))*suma2
    J = -((np.sqrt(k)*g*(4*w + 2*1.j*k))/(np.sqrt(N)*(4*w*w + k*k)))*suma2
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
  sz = 0.5*np.matrix([[1.0, 0.0], [0.0, -1.0]], dtype = complex)
  id = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype = complex)
  sx = np.matrix([[0.0, 1.0], [1.0, 0.0]], dtype = complex)
  sy = np.matrix([[0.0, -1.j], [1.j, 0.0]], dtype = complex)
  
  # Construimos el operador sz
  suma = kronecker(sz, 0, N)
  i = 1
  while(i < N):
    suma += kronecker(sz, i, N)
    i += 1
  spin = suma
  #print(spin)

  # Creamos la base de autoestados de sz
  todo = (q.Qobj(spin)).eigenstates(sparse = False, sort = 'high')
  base = [elemento.full() for elemento in todo[1]]
  #base = [np.array(tup[2], dtype = complex) for tup in todo]
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
  d = ketbra(np.array(ini, dtype = complex), np.array(ini, dtype = complex))
  #res = d/(np.linalg.norm(d))
  return d, ini

# Funcion que me devuelve el operador de espin en una cierta componente y el operador de spin total
def spin_operator(N, component):
    """
    N: int (numero de qubits)
    Component: str (componente del momento angular, 'x', 'y', 'z')
    returns Qobj
    """
    if component == 'x':
        op = q.sigmax()
    elif component == 'y':
        op = q.sigmay()
    elif component == 'z':
        op = q.sigmaz()
    else: 
        raise ValueError('La componente tiene que se x, y, o z')
    
    # Construimos los operadores
    total_operator = 0
    indv_operators = []
    for i in range(N):
        operators = [q.qeye(2) for _ in range(N)]
        operators[i] = op
        indv_operators.append(q.tensor(operators))
        total_operator += q.tensor(operators)
    
    return total_operator, indv_operators

def dicke_bueno(N, params):
    M = 2
    #M = 8
    j = N/2.0
    n = 2*j + 1
    Jz = q.tensor(q.qeye(M), q.jmat(j, 'z'))
    Jx = q.tensor(q.qeye(M), q.jmat(j, 'x'))
    #Jz = q.tensor(q.qeye(M), spin_operator(N, 'z')[0])
    #Jx = q.tensor(q.qeye(M), spin_operator(N, 'x')[0])
    
    sigma = params[0]
    w = params[1]
    k = params[2]
    g = params[3]
    
    
    H = sigma*Jz - ((4.0*w*g*g)/(4.0*w**2 + k**2))*(1.0/N)*Jx*Jx
    J = ((2.0*np.abs(g)*np.sqrt(k))/(np.sqrt(N*(4*w*w + k*k))))*Jx
    #J = -((np.sqrt(k)*g*(4*w + 2*1.j*k))/(np.sqrt(N)*(4*w*w + k*k)))*Jx
    return H, J

def densidad_bueno(N):
    M = 2 # Numero de fotones
    #M = 8
    j = N/2.0
    n = 2*j + 1
    #Jz = q.tensor(q.qeye(M), spin_operator(N, 'z')[0])
    #print(np.allclose(spin_operator(N, 'z')[0], q.jmat(2*j, 'z')))
    Jz = q.tensor(q.qeye(M), q.jmat(j, 'z'))
    #Jz = q.jmat(j, 'z')
    print(Jz.shape)
    # Creamos la base de autoestados de sz
    todo = Jz.eigenstates(sparse = True, sort = 'high')
    base = todo[1]
    #base = [elemento.full() for elemento in todo[1]]
    #base = [np.array(tup[2], dtype = complex) for tup in todo]
    #base = [np.linalg.eig(spin)[1][:, i] for i in range(spin.shape[0])]
    #print(base)

    # Construimos el vector inicial
    a, b = random.random(), random.random()
    ini = (a + b*1.j)*base[0]
    for i in range(1, len(base)):
      a, b = random.random(), random.random()
      ini += (a + b*1.j)*base[i]
    # Le hacemos el ketbra para construir la matriz densidad

    ini = ini.unit()
    d = ini*ini.dag()
    #d = ketbra(ini, ini)
    #res = d/(np.linalg.norm(d))
    return d, ini