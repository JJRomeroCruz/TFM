#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:10:29 2023

@author: juanjo
"""
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import scipy as sc
import random
import time

# Funcion que me hace el producto de kronecker de las matrices identidad y ponemos en la posicion pos una matriz dada
def kronecker(matriz, pos, n):
  id = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype = complex)
  sx = np.matrix([[0.0, 1.0], [1.0, 0.0]], dtype = complex)
  sy = np.matrix([[0.0, -1.j], [1.j, 0.0]], dtype = complex)
  sz = np.matrix([[1.0, 0.0], [0.0, -1.0]], dtype = complex)
  res = id
  for i in range(1, n):
    if(i == pos):
      res = np.kron(res, matriz)
    else:
      res = np.kron(res, id)
  return res

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
  suma2 = kronecker(0.25*np.dot(sx, sx), 0, N)
  suma3 = kronecker(0.5*sx, 0, N)

  i = 1
  while(i < N):
    #print(i)
    suma1 += kronecker(0.5*sz, i, N)
    suma2 += kronecker(0.25*np.dot(sx, sx), i, N)
    suma2 += kronecker(0.5*sx, i, N)
    i += 1

  H = sigma*suma1 - ((4.0*w*g*g)/(4.0*w**2 + k**2))*(1.0/N)*suma2
  J = ((2.0*np.abs(g)*np.sqrt(k))/(np.sqrt(N*(4*w*w + k*k))))*suma3

  return H, J

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

  # Construimos el vector inicial
  a, b = random.random(), random.random()
  ini = (a + b*1.j)*base[0]
  for i in range(1, len(base)):
    a, b = random.random(), random.random()
    ini += (a + b*1.j)*base[i]
  # Le hacemos el ketbra para construir la matriz densidad
  d = ketbra(np.array(ini), np.array(ini))
  #res = d/(np.trace(d))
  res = d/(np.linalg.norm(d))
  return res

# Funcion que me crea una matriz cuadrada de simbolos

def matriz_cuadrada(n):
   matriz = sp.Matrix(n, n, lambda i,j: sp.Symbol('d_(%d)(%d)' % (i,j)))
   return matriz

# Iniciamos la semilla
import random
seed = 3457493
random.seed(a = seed, version = 2)

# Funcion que calcula el hamiltoniano efectivo
def H_eff(H, saltos):
  suma = np.zeros((H.shape[0], H.shape[1]), dtype = complex)
  for salto in saltos:
    suma += salto.H*salto
  Heff = H - 0.5*(1.j)*suma
  return Heff

#Funcion para pasar de matriz a vector espacio de Fock-Liouville
def FL_vector(m):
  m_nuevo = np.reshape(m, m.shape[0]*m.shape[1])
  return sp.Matrix(m_nuevo, dtype = complex)

# Funcion para definir el Limbladiano como matriz en el espacio de Fock-Liouville
def Limblad(H, saltos):
  # Sacamos el hamiltoniano efectivo
  Heff = sp.Matrix(H_eff(H, saltos))

  # Creamos la matriz de simbolos
  d = matriz_cuadrada(H.shape[0])

  # Sacamos la suma de saltos
  suma = sp.zeros(H.shape[0], H.shape[1])
  for salto in saltos:
    suma += salto*d*salto.H

  # Con esto, definimos el Limbladiano
  M = suma - sp.I*(Heff*d - d*Heff.H)
  # Lo pasamos a vector
  M = FL_vector(M)

  # Lo pasamos de sistema de ecuaciones lineal a ecuacion matricial, b tiene que dar el vector de ceros
  #L, b = sp.linear_eq_to_matrix([elemento for elemento in M], [termino for termino in d])
  L, b = sp.linear_eq_to_matrix([M[i] for i in range(len(M))], [d[i] for i in range(len(d))])
  return L, b

# Funcion que calcula las nuevas probabilidades de salto y de no salto
def Calcularp(d, V):
  return np.trace((V.H)*d*V)

# Funcion que nos calcula el nuevo estado tras un salto de tiempo
def NuevoEstado(d, V, H):
  return ((V.H)*d*V)/(np.sqrt(np.trace((V.H)*d*V)))

# Funcionq que resuelve el sistema mediante el metodo de montecarlo
def ResolverSistema(d0, H, salto, N):
  # Establecemos un limite de tiempo
  t = 0.0
  limite = 200.0
  tiempo = [t]
  dt = 0.01

  nfotones = 0 # numero de fotones
  v = [d0]
  # Hamiltoniano efectivo
  Heff = H_eff(H, salto)

  # Operadores de evolución y de salto del sistema
  V = [np.sqrt(dt)*J for J in salto]
  V_0 = kronecker(id, 0, N) - 1.j*dt*Heff

  # Calculamos las probabilides iniciales
  p0 = Calcularp(d0, V_0)
  p = Calcularp(d0, V[0])

  # Empezamos con el bucle
  while(t < limite):
    prob = random.random()
    if(prob < p):
      nfotones += 1
      d0 = NuevoEstado(d0, V[0], H)
    else:
      d0 = NuevoEstado(d0, V_0, H)
    p0 = Calcularp(d0, V_0)
    p = Calcularp(d0, V[0])
    v.append(d0)
    t += dt
    tiempo.append(t)
  return v, tiempo, nfotones

# Funcion que nos hace un plot de lo que queramos
def ploteo_MC(d, tiempo):
  plt.ylabel(r'$\rho_{00}$')
  plt.xlabel('t')

  plt.plot(tiempo, d[0], 'bo', label = '1 ab(sin)')
  plt.plot(tiempo, d[1], 'ro', label = '1 ab(con)')
  #plt.plot(tiempo, d[2], 'ko', label = '2 ab(sin)')
  #plt.plot(tiempo, d[3], 'yo', label = '2 ab(con)')
  #plt.plot(tiempo, d[4], 'b--', label = '1 cerr(sin)')
  #plt.plot(tiempo, d[5], 'r--', label = '1 cerr(con)')
  #plt.plot(tiempo, d[6], 'k--', label = '2 cerr(sin)')
  #plt.plot(tiempo, d[7], 'y--', label = '2 cerr(con)')

  plt.legend(loc = 'upper right')
  plt.show()

  return

# Funcion que nos saca la posicion del segundo valor mas grande en una lista
def segunda_posicion_mas_grande(lista):
    if len(lista) < 2:
        return "La lista no tiene suficientes elementos"

    # Encontrar el elemento más grande y su índice
    maximo = max(lista)
    indice_maximo = lista.index(maximo)

    # Eliminar el elemento más grande de la lista
    lista.remove(maximo)

    # Encontrar el segundo elemento más grande y su índice
    segundo_maximo = max(lista)
    indice_segundo_maximo = lista.index(segundo_maximo)

    # Ajustar el índice para tener en cuenta la eliminación del máximo anterior
    if indice_segundo_maximo >= indice_maximo:
        indice_segundo_maximo += 1

    return indice_segundo_maximo

# Funcion que nos obtiene la transformacion unitaria del efecto Mpemba a partir del lindbladiano
def Mpemba(L, d, N):
  # Diagonalizamos el lindbladiano
  # Sacamos los autovalores y los autovectores del Limbladiano
  """
  todo = (L.H).eigenvects()
  vals = [todo[i][0] for i in range(L.shape[0])]
  """
  L_H = np.matrix(L.H, dtype = complex)
  vals, L_e = np.linalg.eig(L_H)
  #vals = list(sp.simplify(L.eigenvals()).keys())
  """L_e = [sp.Matrix(tup[2][0]) for tup in todo]"""
  #L_e = [sp.Matrix(tup[2][0]) for tup in (L.H).eigenvects()]
  #vals, L_e = np.linalg.eig(np.matrix(L.H, dtype = complex))
  """vals = list(sp.re(vals[i]) for i in range(len(vals)))"""
  print('Hemos diagonalizado el lindbladiano')
  # Normalizamos las automatrices
  L_e = [np.reshape(matriz, (d.shape[0], d.shape[1])) for matriz in L_e]
  L_e = [matriz.T/(np.sqrt((np.trace((np.dot(((matriz.conj()).T), matriz)))))) for matriz in L_e]
  #L_e = [matriz/matriz.norm() for matriz in L_e]
  #vals, L_e = np.linalg.eig(np.matrix(L.H, dtype = complex))
  #vals = list(vals)
  # Extraemos la matriz por la izquierda cuyo autovalor es el mayor (y no es cero)
  lista_vals =  list(-np.real(vals))
  maximo = max(lista_vals)
  indice_maximo = lista_vals.index(maximo)
  lista_vals.remove(maximo)
  segundo_maximo = max(lista_vals)
  indice_segundo_maximo = lista_vals.index(segundo_maximo)
  if indice_segundo_maximo >= indice_maximo:
    indice_segundo_maximo += 1
    
  print("Dimesiones de la matriz densidad: " + str(d.shape))
  print("Dimensiones del autovector L_e: " + str(L_e[indice_segundo_maximo].shape))
  
  #L1 = sp.Matrix(np.reshape(L_e[indice_segundo_maximo], (d.shape[0], d.shape[1])), dtype = complex)
  L1 = np.reshape(L_e[indice_segundo_maximo], (d.shape[0], d.shape[1]))
  
  #L1 = np.reshape(L_e[indice_segundo_maximo], (d.shape[0], d.shape[1]))
  print('Hemos obtenido L1')
  print("Dimension de L1: " + str(L1.shape))
  print(L1)
  # Diagonalizamos L1 y sacamos sus autovalores
  #todo1 = (L1.H).eigenvects()
  #vals = [todo1[i][0] for i in range(L1.shape[0])]
  #vects = [sp.Matrix(tup[2][0]) for tup in todo1]
  #vals = list(vals)
  vals, vects = np.linalg.eig(L1)
  vals = list(vals)
  # Normalizamos los autovectores de la matriz L1
  #vects = [matriz/matriz.norm() for matriz in vects]
  vects = [matriz/np.linalg.norm(matriz) for matriz in vects]
  #vects = [matriz/(np.dot(np.conjugate(matriz.T), matriz)) for matriz in vects]
  # Ahora, tenemos que dividir en dos caminos, definimos una tolerancia
  tol = 1.0e-5
  print(vals)
  es_cero = [np.abs(vals[i]) < tol for i in range(len(vals))]
  print(es_cero)

  if(any(es_cero)):
    print('Hemos elegido la via del cero')
    # Nos vamos al caso en el que un autovalor es 0
    indice = es_cero.index(True)
    #U = ketbra(np.array(vects[0].T), np.array(vects[indice].T))
    U = ketbra(np.array(vects[0].T), np.array(vects[indice].T))
  else:
    print('Hemos elegido la via del no cero')
    # Si no hay ningun autovalor que sea 0, se coje una pareja de autovalores con signo contrario
    i = 0
    indice_contrario = 0
    while(i < len(vals)):
      if(vals[i]*vals[0] < 0):
        indice_contrario = i
      else:
        i += 1
    # Ahora, construimos la transformacion unitaria
    F = ketbra(vects[0], vects[indice_contrario]) + ketbra(vects[indice_contrario], vects[0])
    s = np.arctan(np.sqrt((np.abs(vals[0]))/(np.abs(vals[indice_contrario]))))
    identidad = kronecker(id, 0, N)
    U = identidad + (np.cos(s) - 1.0)*(np.dot(F, F)) - 1.j*F

  return U

""" Ahora, empezamos usando las funciones para crear la transformacion unitaria """

dt = 0.01 #salto de tiempo
#sigma = 1.0
#k = 4.0*sigma

# Medimos el tiempo
t1 = time.time()

# Matrices de Pauli
id = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype = complex)
sx = np.matrix([[0.0, 1.0], [1.0, 0.0]], dtype = complex)
sy = np.matrix([[0.0, -1.j], [1.j, 0.0]], dtype = complex)
sz = np.matrix([[1.0, 0.0], [0.0, -1.0]], dtype = complex)
num = np.matrix([[1.0, 0.0], [0.0, 0.0]], dtype = complex)

# Hamiltoniano, operador salto y hamiltoniano efecivo
N = 5
sigma = 1.0

#w = 0.25*sigma
w = 1.0*sigma
#k = 0.5*sigma + 0.5*np.sqrt(1-8*w*w)
#k = 0.5*sigma
k = 1.0*sigma
g = 1.0*sigma
params = [sigma, w, k, g]
H, J = dicke(N, params)
Heff = H_eff(H, [J])
print(Heff.shape)

# Matriz densidad
d0 = densidad(N)
print('densidad hecha')
L, b = Limblad(sp.Matrix(H, dtype = complex), [sp.Matrix(J, dtype = complex)])
print(L.shape)
U = Mpemba(L, d0, N)

duracion = time.time() - t1

# Pasamos la U a un fichero
# Creamos en fichero
#f = open('Mpemba.dat', 'x')
#f.close()

def guardar_matriz_en_archivo(matriz, nombre_archivo):
    try:
        with open(nombre_archivo, 'w') as archivo:
            for fila in matriz:
                fila_str = ' '.join(map(str, fila))
                archivo.write(fila_str + '\n')
        print(f"Matriz guardada exitosamente en {nombre_archivo}")
    except Exception as e:
        print(f"Error al guardar la matriz en el archivo: {e}")

nombre_archivo_salida = 'Mpemba.dat'
guardar_matriz_en_archivo(U, nombre_archivo_salida)


d0_exp = np.dot(np.dot(U, d0), np.conjugate(U.T))
print(d0.shape)
v1, tiempo, n1 = ResolverSistema(d0, H, [J], N)
v2, tiempo, n2 = ResolverSistema(d0_exp, H, [J], N)

dens = [v1, v2]

ob = [[np.real(v[i][0, 0]) for i in range(len(v))] for v in dens]
print(n1)
print(n2)
#plt.ylim(0.009, 0.014)
#plt.xlim(0, 10)
plt.plot(tiempo, ob[0], 'bo')
plt.plot(tiempo, ob[1], 'ro')
plt.show()
print(f"He tardado {duracion} segundos.")