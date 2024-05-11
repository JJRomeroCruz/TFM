#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:25:36 2023

@author: juanjo
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy as sc
import random

"Aqui voy a definir las funciones que he estado usando para la simulacion numerica mediante el quantum montecarlo jump"
# Funcion que calcula el hamiltoniano efectivo
def H_eff(H, saltos):
  suma = np.zeros((H.shape[0], H.shape[1]), dtype = complex)
  for salto in saltos:
    suma += salto.H*salto
  Heff = H - 0.5*np.complex(0, 1)*suma
  return Heff

# Funcion que calcula las nuevas probabilidades de salto y de no salto
def Calcularp(d, V):
  return np.trace((V.H)*d*V)

# Funcion que nos calcula el nuevo estado tras un salto de tiempo
def NuevoEstado(d, V):
  return ((V.H)*d*V)/(np.sqrt(np.trace((V.H)*d*V)))

def ResolverSistema(d0, H, salto):
  # Establecemos un limite de tiempo y un salto de tiempo
  dt = 0.0001
  t = 0.0
  limite = 5.0
  tiempo = [t]

  v = [d0]
  # Hamiltoniano efectivo
  Heff = H_eff(H, salto)

  # Operadores de evolución y de salto del sistema
  V = np.sqrt(dt)*salto
  V_0 = id - np.complex(0, 1)*dt*Heff

  # Calculamos las probabilides iniciales
  p0 = Calcularp(d0, V_0)
  p = Calcularp(d0, V)

  # Empezamos con el bucle
  while(t < limite):
    prob = random.random()
    if(prob < p):
      d0 = NuevoEstado(d0, V)
    else:
      d0 = NuevoEstado(d0, V_0)
    p0 = Calcularp(d0, V_0)
    p = Calcularp(d0, V)
    v.append(d0)
    t += dt
    tiempo.append(t)
  return v, tiempo