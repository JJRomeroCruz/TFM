#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:14:39 2023

@author: juanjo
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy as sc
import random

"Aquí vamos a recopilar las funciones que hemos usado para el estudio de sistemas cuánticos abiertos "

"Para las resolucion analítica del sistema"
# Funcion para obtener el Hamiltoniano efectivo, para poner el hermitico basta con poner m.H
def H_eff(H, saltos):
  suma = sp.zeros(H.shape[0], H.shape[1])
  for salto in saltos:
    suma += (salto.H)*salto
  Heff = H - 0.5*sp.I*suma

  return Heff

#Funcion para pasar de matriz a vector espacio de Fock-Liouville
def FL_vector(m):
  m_nuevo = np.reshape(m, m.shape[0] + m.shape[1])
  return sp.Matrix(m_nuevo)

# Funcion para definir el Limbladiano como matriz en el espacio de Fock-Liouville
def Limblad(H, saltos):
  # Sacamos el hamiltoniano efectivo
  Heff = H_eff(H, saltos)

  # Definimos la matriz densidad y la pasamos a vector en el espacio F-L
  d_00, d_01, d_10, d_11 = sp.symbols('d_00, d_01, d_10, d_11')
  d = sp.Matrix([[d_00, d_01], [d_10, d_11]])

  # Sacamos la suma de saltos
  suma = sp.zeros(H.shape[0], H.shape[1])
  for salto in saltos:
    suma += salto*d*salto.H

  # Con esto, definimos el Limbladiano
  M = suma - sp.I*(Heff*d - d*Heff.H)

  # Lo pasamos a vector
  M = FL_vector(M)

  # Lo pasamos de sistema de ecuaciones lineal a ecuacion matricial, b tiene que dar el vector de ceros
  L, b = sp.linear_eq_to_matrix([termino for termino in M], [termino for termino in d])
  return L, b


# Funcion que me saca la evolucion temporal de la matriz densidad para un valor inicial
def espectral(d_0, L):

  # Sacamos los autovalores y los autovectores del Limbladiano
  vals = list(sp.simplify(L.eigenvals()).keys())
  R_e = [sp.Matrix(tup[2][0]) for tup in L.eigenvects()]
  L_e = [sp.Matrix(tup[2][0]) for tup in (L.H).eigenvects()]

  # Normalizamos las automatrices
  R_e = [matriz/matriz.norm() for matriz in R_e]
  L_e = [matriz/matriz.norm() for matriz in L_e]

  # Pasamos la matriz densidad inicial a vector en el espacio de F-L
  d_0_vector = FL_vector(d_0)

  suma = sp.zeros(1, d_0_vector.shape[0])
  t = sp.Symbol('t', Real = True)
  for i in range(len(vals)):
    ter1 = L_e[i].T*d_0_vector
    ter = (sp.exp(t*vals[i])*ter1)*R_e[i].T
    suma += ter
  return sp.Matrix(np.reshape(suma, (d_0.shape[0], d_0.shape[1])))

def valor_esperado(den, A):
  return (sp.Trace(A*den)).simplify()
