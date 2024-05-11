#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:31:15 2023

@author: juanjo
"""
import numpy as np
import sympy as sp
import scipy as sc
import matplotlib.pyplot as plt

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
def espectral1(d_0, L):

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

def espectral2(d_0, L):
  L = np.matrix(L, dtype = complex)

  # Sacamos los autovalores y los autovectores del Limbladiano
  vals, R_e = sc.linalg.eig(L, right = True)
  vals, L_e = sc.linalg.eig(L.H, right = True)
  R_e = [sp.Matrix(matriz) for matriz in R_e]
  L_e = [sp.Matrix(matriz) for matriz in L_e]
  #vals = list(sp.simplify(L.eigenvals()).keys())
  #R_e = [sp.Matrix(tup[2][0]) for tup in L.eigenvects()]
  #L_e = [sp.Matrix(tup[2][0]) for tup in (L.H).eigenvects()]

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

# Matrices de pauli y la identidad
id = sp.Matrix([[1.0, 0.0], [0.0, 1.0]])
sx = sp.Matrix([[0.0, 1.0], [1.0, 0.0]])
sy = sp.Matrix([[0.0, -sp.I], [sp.I, 0.0]])
sz = sp.Matrix([[1.0, 0.0], [0.0, -1.0]])
n = sp.Matrix([[0.0, 0.0], [0.0, 1.0]])
d = sp.Matrix([[1.0, 2.0], [3.0, 4.0]])

# El hamiltoniano, el operador de salto y diferentes cosas
# De momento, vamos a hacer para sigma = 1 y k = 4
w, k = sp.symbols('w, k', Real = True)
t = sp.Symbol('t', Real = True)
sigma = 1.0
ka = 4.0*sigma
H = sigma*sx
J = sp.sqrt(ka)*sp.Matrix([[0.0, 0.0], [1.0, 0.0]])

d01 = sp.Matrix([[9.0/20.0, 0.25*sp.I], [-sp.I*0.25, 11.0/20.0]])

# Limbladiano sistema abierto
L_ab, b_ab = Limblad(H, [J])
# Limbladiano sistema cerrado
L_cerr, b_cerr = Limblad(H, [sp.zeros(H.shape[0], H.shape[1])])

# Resolvemos el sistema para ambos metodos
d_ab1 = espectral1(d01, L_ab)
d_ab2 = espectral2(d01, L_ab)

d_cerr1 = espectral1(d01, L_cerr)
d_cerr2 = espectral2(d01, L_cerr)

# Representamos todo
tiempo = np.linspace(0.0, 5.0, 50)

y1 = [sp.re(d_ab1[1, 1]).subs(t, salto) for salto in tiempo]
y2 = [(sp.re(d_ab2[1, 1])).subs(t, salto) for salto in tiempo]
y4 = [sp.re(d_cerr1[1, 1]).subs(t, salto) for salto in tiempo]
y5 = [(sp.re(d_cerr2[1, 1])).subs(t, salto) for salto in tiempo]

plt.title("Parte real")
plt.ylabel(r'$\rho_{00}$')
plt.xlabel('t')

plt.plot(tiempo, y1, 'bo', label = '1 ab')
plt.plot(tiempo, y2, 'yo', label = '2 ab')
plt.plot(tiempo, y4, 'b--', label = '1 cerr')
plt.plot(tiempo, y5, 'y--', label = '2 cerr')

plt.legend(loc = 'upper right')
plt.show()

plt.savefig('rho_00.png', format = 'png')

print(sp.re(d_ab2[1, 1]).subs(t, 3.0))