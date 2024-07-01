#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:41:35 2024

@author: juanjo
"""
import numpy as np
import sympy as sp
import scipy as sc
import matplotlib.pyplot as plt
import random as random
import qutip as q
from matplotlib import cm
import dicke

# Funcion que me hace el producto de kronecker de las matrices identidad y ponemos en la posicion pos una matriz dada
def kronecker(matriz, pos, n):
  ide = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype = complex)
  if(pos == 0):
    res = matriz
  else:
    res = ide

  for i in range(1, n):
    if(i == pos):
      res = np.kron(res, matriz)
    else:
      res = np.kron(res, ide)
  return res

# Funcion que me hace el ketbra de dos vectores
def ketbra(v1, v2):
  m = np.outer(v1, np.conjugate(v2.T))
  return m

# Funcion que me crea una matriz cuadrada de simbolos
def matriz_cuadrada(n):
   matriz = sp.Matrix(n, n, lambda i,j: sp.Symbol('d_(%d)(%d)' % (i,j)))
   return matriz

# Funcion que calcula el hamiltoniano efectivo
def H_eff(H, saltos):
    suma = q.Qobj(np.zeros_like(H, dtype = 'complex'))
    for salto in saltos:
        suma += (salto.dag())*salto
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
  saltos_sp = [sp.Matrix(elemento, dtype = complex) for elemento in saltos]
  for salto in saltos_sp:
      suma += (salto*d)*salto.H
      #suma += np.dot(np.dot(salto, d), salto.H)

  # Con esto, definimos el Limbladiano
  M = suma - sp.I*(Heff*d - d*Heff.H)
  #M = sp.Matrix(suma, dtype = complex) - sp.I*(np.dot(Heff, d) - np.dot(d, Heff.H))
  # Lo pasamos a vector
  M = FL_vector(M)

  # Lo pasamos de sistema de ecuaciones lineal a ecuacion matricial, b tiene que dar el vector de ceros
  #L, b = sp.linear_eq_to_matrix([elemento for elemento in M], [termino for termino in d])
  L, b = sp.linear_eq_to_matrix([M[i] for i in range(len(M))], [d[i] for i in range(len(d))])
  return L, b

# Funcion que calcula las nuevas probabilidades de salto y de no salto
def Calcularp(d, V):
    return (V*d*V.dag()).tr()

# Funcion que nos calcula el nuevo estado tras un salto de tiempo
def NuevoEstado(d, V, H):
    num = V*d*V.dag()
    den = np.sqrt((V*d*V.dag()).tr())
    return num/den

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
def Mpemba1(L, L_e, autovals, d, N, ini):

  # Extraemos la matriz por la izquierda cuyo autovalor es el mayor (y no es cero)
  lista_vals =  list(-np.real(autovals))
  maximo = max(lista_vals)
  indice_maximo = lista_vals.index(maximo)
  lista_vals.remove(maximo)
  segundo_maximo = max(lista_vals)
  indice_segundo_maximo = lista_vals.index(segundo_maximo)
  if indice_segundo_maximo >= indice_maximo:
    indice_segundo_maximo += 1

  L1 = np.reshape(L_e[indice_segundo_maximo], (d.shape[0], d.shape[1]))

  todo = ((sp.Matrix(L1, dtype = complex)).H).eigenvects()
  vals = [tup[0] for tup in todo]
  vects = np.asarray([tup[2] for tup in todo], dtype = complex)
  #vals, vects = np.linalg.eig(L1)
  vals = list(np.array(vals, dtype = complex))
  print('Autovalores de L1: ')
  print(vals)

  # Normalizamos los autovectores de la matriz L1
  #vects = [matriz/np.linalg.norm(matriz) for matriz in vects]

  # Ahora, tenemos que dividir en dos caminos, definimos una tolerancia
  tol = 1.0e-14
  es_cero = [(np.abs(np.real(vals[i])) < tol) for i in range(len(vals))]

  if(any(es_cero)):
    print('Se puede hacer la vía del cero, con el autovalor: ')
    # Nos vamos al caso en el que un autovalor es 0
    indice = es_cero.index(True)
    print(vals[indice])
    print("Autovector con shape: " + str(vects[indice][0].shape) + ' es: ' + str(vects[indice][0]))
    U = ketbra(ini, vects[indice][0])
  else:
    print('No se ha podido hacer la via del cero')
    # Si no hay ningun autovalor que sea 0, se coje una pareja de autovalores con signo contrario
    U = np.zeros(d.shape)
  return U

def eliminar_duplicados(lista):
    # Convierte la lista a un conjunto para eliminar duplicados
    conjunto_sin_duplicados = set(lista)

    # Convierte el conjunto de nuevo a una lista
    lista_sin_duplicados = list(conjunto_sin_duplicados)

    return lista_sin_duplicados

def Mpemba2(L, L_e, vals, d, N, ini):

  # Extraemos la matriz por la izquierda cuyo autovalor es el mayor (y no es cero)
  lista_vals =  list(-np.real(np.array(vals, dtype = complex)))
  maximo = max(lista_vals)
  print('Maximo: ' + str(maximo))
  indice_maximo = lista_vals.index(max(lista_vals))
  lista_vals.remove(maximo)
  segundo_maximo = max(lista_vals)
  indice_segundo_maximo = lista_vals.index(segundo_maximo)
  if indice_segundo_maximo >= indice_maximo:
    indice_segundo_maximo += 1

  L1 = np.reshape(L_e[indice_segundo_maximo], (d.shape[0], d.shape[1]))
  L1 = sp.Matrix(L1, dtype = complex)

  todo = L1.eigenvects()
  autovals = [tup[0] for tup in todo]
  autovects = [np.array(tup[2][0], dtype = complex) for tup in todo]
  autovals = list(np.array(autovals, dtype = complex))
  print('Autovalores de L1: ')
  print(autovals)

  # Ahora, tenemos que dividir en dos caminos, definimos una tolerancia
  tol = 1.0e-10 # N = 5
  autovals = eliminar_duplicados(autovals)
  es_cero = [(np.abs(np.real(autovals[i])) < tol) for i in range(len(autovals))]

  print('Vamos a probar la via del no cero')
  # Se coje una pareja de autovalores con signo contrario
  i = 0
  indice_contrario = 0
  indice_inicial = es_cero.index(False)
  autovals = eliminar_duplicados(autovals)
  es_contrario = [np.real(autovals[indice_inicial])*np.real(autovals[i]) < 0 for i in range(len(autovals))]
  if(any(es_contrario)):
    indice_contrario = es_contrario.index(True)
    print('(indice contrario, indice inicial, len(autovects)) = ' + str(indice_contrario) + ', ' + str(indice_inicial) + ', '  + ' ' + str(autovects[0].shape))
    F = ketbra(autovects[indice_inicial], autovects[indice_contrario]) + ketbra(autovects[indice_contrario], autovects[indice_inicial])
    s = np.arctan(np.sqrt((np.abs(autovals[indice_inicial]))/(np.abs(autovals[indice_contrario]))))
    print("La s me sale: " + str(s) + ' se ha cogido la s que sale de los autovalores: ')
    print((autovals[indice_inicial], autovals[indice_contrario]))
    identidad = kronecker(id, 0, N)
    U = identidad + (np.cos(s) - 1.0)*(np.dot(F, F)) - 1.j*F
  else:
    print('No se puede coger la vía del no cero')
    U = np.zeros(d.shape)
  return U

def indices_mayores_que_100(matriz, n):
    indices = []
    filas = matriz.shape[0]
    columnas = matriz.shape[1]

    for i in range(filas):
        for j in range(columnas):
            if(np.real(matriz[i, j]) > n):
                indices.append((i, j))

    return indices

# Funcion que nos saca el estado estacionario
def estacionario(L, d):
  # Sacamos los autovalores y los autovectores del Limbladiano
  todo = L.eigenvects()
  vals = [tup[0] for tup in todo]
  vects = [np.array(tup[2], dtype = complex) for tup in todo]

  # Sacamos el estado estacionario que es el autovector correspondiente al autovalor mas cercano a 0
  vals = np.array(vals, dtype = complex)
  espositivo = False
  for elemento in vals:
    if(elemento > 0.0):
      indice = list(vals).index(elemento)
      espositivo = True
  if(espositivo == False):
    reales = [np.real(valor) for valor in np.array(vals, dtype = complex)]
    minimo = max(reales)
    indice = reales.index(minimo)

  # Pasamos el estado estacionario a matriz
  r = np.reshape(vects[indice], (d.shape[0], d.shape[1]))

  if(np.imag(np.trace(r)) < 1e-10):
    res = r/np.trace(np.real(r))
  else:
    res = r/np.trace(r)
  return res

# Funcion que resuelve el sistema mediante el metodo de montecarlo
def Resolver_Sistema(d0, H, salto, N, nveces, limite_t):
    dt = 0.01
    tiempo = np.linspace(0, limite_t, int(limite_t/dt))
    vector_fotones = []
    nfotones = 0
    Heff = H_eff(H, salto)
    
    # Operadores de salto
    V = [np.sqrt(dt)*J for J in salto]
    V0 = q.qeye(Heff.shape[0]) - 1.j*dt*Heff
    intentos = []
    
    # Empezamos con el bucle
    for i in range(nveces):
        nfotones = 0
        d = d0
        res = []
        # Probabilidades iniciales
        p0 = Calcularp(d0, V0)
        p = Calcularp(d0, V[0])
        #p = 1-p0
        for t in tiempo:
            prob = random.random()
            #print('probabilidad: ', p)
            if(prob < np.abs(p)):
                nfotones += 1
                d = NuevoEstado(d, V[0], H)
            else:
                d = NuevoEstado(d, V0, H)
            
            # Calculamos las nuevas probabilidades
            p0 = Calcularp(d, V0)
            #p = 1-p0
            p = Calcularp(d, V[0])
            res.append(d)
        intentos.append(res)
        vector_fotones.append(nfotones)
        
    # Calculamos el promedio de las trayectorias
    suma = [0 for i in range(len(intentos[0]))]
    for i in range(len(intentos[0])):
        for elemento in intentos:
            suma[i] += elemento[i]
    final = [elemento/nveces for elemento in suma]
    return final, tiempo, vector_fotones
    
    
# Funcion que, dado un vector y su dimension, me genera una base de vectores ortonormales en la cual este mismo vector es el primero
def generar_base_ortonormal(vector, dim):
    # Normalizar el primer vector
    v1 = vector / np.linalg.norm(vector)
    v1 = np.squeeze(np.asarray(v1, dtype = complex))
    # Generar una matriz aleatoria de dimensión (dim x dim)
    random_matrix = np.matrix(np.random.randn(dim, dim), dtype = complex)

    # Realizar el proceso de Gram-Schmidt para generar los vectores ortogonales restantes
    base = [v1]
    for i in range(1, dim):
        # Obtener el siguiente vector aleatorio
        #v = np.reshape(random_matrix[i], (1, max(random_matrix[i].shape)))
        v = np.squeeze(np.asarray(random_matrix[i], dtype = complex))
        #v = np.asarray(random_matrix[i], dtype = complex)
        #v = np.reshape(v, (1, max(v.shape)))
        # Restar las proyecciones de los vectores anteriores
        for j in range(i):
            v -= np.dot(v, base[j].T) * base[j]

        # Normalizar el vector resultante
        v /= np.linalg.norm(v)

        # Agregar el vector a la base ortonormal
        base.append(v)

    return base

# Funcion que me encuentra el segundo maximo
def buscar_segundo_maximo(lista):
    # Si la lista tiene menos de dos elementos, no hay segundo máximo
    if len(lista) < 2:
        return None

    # Inicializamos el máximo y el segundo máximo con los dos primeros elementos de la lista
    maximo = max(lista[0], lista[1])
    segundo_max = min(lista[0], lista[1])

    # Iteramos sobre los elementos restantes de la lista para encontrar el máximo y el segundo máximo
    for num in lista[2:]:
        if num > maximo:
            segundo_max = maximo
            maximo = num
        elif num > segundo_max and num != maximo:
            segundo_max = num
    return segundo_max, lista.index(segundo_max)

# Funcion que nos calcula la transformacion de Mpemba2, pero teniendo en cuenta la matriz de cambio de base
def Mpemba2_mejorada(L, L_e, vals, d, N, ini):
  iden = np.matrix([[1.0, 0.0], [0.0, 1.0]], dtype = complex)
  
  # Extraemos la matriz por la izquierda cuyo autovalor es el mayor (y no es cero)
  
  # segundo_maximo, indice_segundo_maximo = buscar_segundo_maximo(list(np.real(vals)))
  segundo_maximo, indice_segundo_maximo = buscar_segundo_maximo([np.real(elemento) for elemento in vals])
  
  # La pasamos a matriz
  L1 = np.reshape(L_e[indice_segundo_maximo], (d.shape[0], d.shape[1]))

  # Diagonalizamos la matriz L1
  todo = (q.Qobj(L1)).eigenstates(sparse = False, sort = 'low')
  autovals = todo[0]
  autovects = [elemento.full() for elemento in todo[1]]

  # Ahora, generamos la base auxiliar para el vector estado inicial
  base_aux = generar_base_ortonormal(ini, int(2**N))

  base_aux = np.array([elemento for elemento in base_aux], dtype = complex)
  
  # Con esto, podemos generar la primera transformacion
  U_cambio = ketbra(autovects[0], base_aux[0])
  for i in range(1, N):
    U_cambio += ketbra(autovects[i], base_aux[i])

  es_cero = [(np.allclose(np.abs(autovals[i]), 0, atol = 1e-5)) for i in range(len(autovals))]

  print('Vamos a probar la via del no cero')
  # Se coje una pareja de autovalores con signo contrario
  i = 0
  indice_contrario = 0
  indice_inicial = es_cero.index(False)
  print(indice_inicial)
  es_contrario = [np.real(autovals[indice_inicial])*np.real(autovals[i]) < 0 for i in range(len(autovals))]
  if(any(es_contrario)):
    indice_contrario = es_contrario.index(True)
    F = ketbra(autovects[indice_inicial], autovects[indice_contrario]) + ketbra(autovects[indice_contrario], autovects[indice_inicial])
    s = np.arctan(np.sqrt((np.abs(autovals[indice_inicial]))/(np.abs(autovals[indice_contrario]))))
    print("La s me sale: " + str(s) + ' se ha cogido la s que sale de los autovalores: ')
    identidad = kronecker(iden, 0, N)
    U = identidad + (np.cos(s) - 1.0)*(np.dot(np.conjugate(F.T), F)) - 1.j*np.sin(s)*F
  else:
    print('No se puede coger la vía del no cero')
    U = np.zeros(d.shape)
  return q.Qobj(U), U_cambio

def estacionario_bueno(vals, vects_r, vects_l, d):
  vals_copia = list(vals[:])
  # Sacamos los autovalores que son 0
  estacionarios = [num for num in vals_copia if np.allclose(np.abs(num), 0.0)]
    
  # Sacamos los indices de esos autovalores en la lista vals
  indices_est = [vals_copia.index(num) for num in estacionarios]
  # Sacamos los autovectores correspondientes y le hacemos reshape
  m_est = [np.reshape(vects_r[indice], d.shape) for indice in indices_est]
  m_est = [np.trace(np.dot(vects_l[indices_est[0]], d))*m_est[i] for i in range(len(m_est))]

  return m_est

# Funcion que genera la transformacion de Mpemba1, pero haciendo antes lo de la matriz de cambio de base
def Mpemba1_mejorada(L, L_e, autovals, d, N, ini):

  # Se obtiene el segundo autovalor con la parte real mayor
  segundo_maximo, indice_segundo_maximo = buscar_segundo_maximo(list(np.real(autovals)))

  # Hacemos reshape
  L1 = np.reshape(L_e[indice_segundo_maximo], (d.shape[0], d.shape[1]))

  # Diagonalizamos la matriz L1
  todo = (q.Qobj(L1)).eigenstates(sparse = False, sort = 'low')
  vals = todo[0]
  vects = [elemento.full() for elemento in todo[1]]
  
  # Ahora, generamos la base auxiliar para el vector ini
  base_aux = generar_base_ortonormal(ini, int(2**N))
  base_aux = np.asarray([elemento for elemento in base_aux], dtype = complex)
  U_cambio = ketbra((vects[0]), base_aux[0])
  for i in range(1, N):
    U_cambio += ketbra((vects[i]), base_aux[i])

  # Ahora, tenemos que dividir en dos caminos, definimos una tolerancia
  es_cero = [(np.allclose(np.abs(vals[i]), 0.0, atol = 1e-2)) for i in range(len(vals))]

  if(any(es_cero)):
    print('Se puede hacer la vía del cero, con el autovalor: ')
    # Nos vamos al caso en el que un autovalor es 0
    indice = es_cero.index(True)
    print(vals[indice])
    U = ketbra(ini, vects[indice])
  else:
    print('No se ha podido hacer la via del cero')
    # Si no hay ningun autovalor que sea 0, se coje una pareja de autovalores con signo contrario
    U = np.zeros(d.shape)
  return q.Qobj(U), U_cambio

# Funcion que nos construye la transformacion de Mpemba, pero a base de transformaciones a un solo qubit
def Mpemba_sep(theta, phi, N):
    iden = q.qeye(int(2**N))
    spin_z = dicke.spin_operator(N, component = 'z')[1]
    spin_y = dicke.spin_operator(N, component = 'y')[1]
    #spin_z = [q.tensor(q.qeye(M), elemento) for elemento in dicke.spin_operator(N, 'z')[1]]
    #spin_y = [q.tensor(q.qeye(M), elemento) for elemento in dicke.spin_operator(N, 'y')[1]]
    res = iden.full()
    #print(iden.shape, spin_z[0].shape)
    
    for i in range(N):
        U1 = np.cos(0.5*phi)*iden.full() + 1.j*np.sin(0.5*phi)*spin_z[i].full()
        U2 = np.cos(0.5*theta)*iden.full() + 1.j*np.sin(0.5*theta)*spin_y[i].full()
        res = np.dot(np.dot(U1, U2), res)
    return q.Qobj(res)

# Funcion que da la evolución temporal de la diagonalizacion del lindbladiano
def solucion(d, r, l, autovals, tiempo):
  res = []
  suma = 0.0
  #tol = 1e-16
  for t in tiempo:
    suma = 0.0
    #suma = r[0]*np.trace(np.dot(l[0], d))*np.exp(t*autovals[0])
    for i in range(0, len(r)):
        if(np.real(autovals[i]) < 0):
            suma += r[i]*((l[i]*d).tr())*np.exp(t*autovals[i])
            #suma += r[i]*np.trace(np.dot(l[i], d))*np.exp(t*autovals[i])
    res.append(suma)
  return res

# Funcion que me busca los angulos que me permiten acelerar el decaimiento
def buscar_angulos(L1, d0, N):
    epsilon = np.abs(np.trace(np.dot(L1, d0)))
    #epsilon = np.abs((L1*d0).tr())
    #epsilon = 1e-2
    posibles = []
    phi = 0.0
    theta = 0.0
    traza = []
    
    while(theta < np.pi):
        phi = 0.0
        while(phi < 2.0*np.pi):
            U = Mpemba_sep(theta, phi, N)
            #new_rho = U*d0*U.dag()
            new_rho = np.dot(U.full(), np.dot(d0, np.conjugate((U.full()).T)))
            res = np.trace(np.dot(L1, new_rho))
            #res = (L1*new_rho).tr()
            if(np.abs(res) < epsilon):
                posibles.append([theta, phi])
                traza.append(np.abs(res))
            phi += 0.1
        theta += 0.1
    
    return posibles, traza

# Funcion que me obtiene el estado estacionario con qutip (a ver si esta es la buena)
def estacionario_q(H, list_J):
    r = q.steadystate(q.Qobj(H), [q.Qobj(J) for J in list_J], sparse = False)
    return r.full()

# Funcion que representa la esfera parametrizada segun angulos
def parametrizacion_esfera(r, theta_range, phi_range):
    theta_values = np.linspace(*theta_range, num = 100)
    phi_values = np.linspace(*phi_range, num = 100)
    
    theta, phi = np.meshgrid(theta_values, phi_values)
    
    x = r * np.sin(theta)*np.cos(phi)
    y = r* np.sin(theta)*np.sin(phi)
    z = r * np.cos(theta)
    
    return x, y, z

# Funcion que representa en la esfera los valores de los angulos que yo quiera
def esfera_partes(r, theta_values, phi_values):
    theta, phi = np.meshgrid(theta_values, phi_values)
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z

# Funcion que me construye el producto tensorial de matrices identidad con una matriz en la posicion que yo quiera
def kronecker_q(matriz, pos, N):
    ide = q.qeye(2)
    if(pos == 0):
      res = matriz
    else:
      res = ide

    for i in range(1, N):
      if(i == pos):
        res = q.tensor(res, matriz)
      else:
        res = q.tensor(res, ide)
    return res

# Funcion que me genera una base ortonormal en la cual esta incluido un vector que yo le paso
def generar_base_ortonormal_q(vector):
    # Convertir el vector a un objeto Qobj
    vector_qobj = q.Qobj(vector)

    # Normalizar el vector
    vector_normalizado = vector_qobj.unit()

    # Definir el primer vector de la base como el vector normalizado
    base = [vector_normalizado]

    # Generar los vectores restantes de la base utilizando Gram-Schmidt
    for i in range(len(vector.full())):
        if i == 0:
            continue
        # Proyectar el vector original sobre los vectores de la base ya generados
        proyecciones = [vector_normalizado.overlap(v) * v for v in base]
        # Restar las proyecciones del vector original para ortogonalizarlo
        vector_ortogonalizado = vector_qobj - sum(proyecciones)
        # Normalizar el vector ortogonalizado para obtener un nuevo vector de la base
        vector_normalizado = vector_ortogonalizado
        # Agregar el nuevo vector a la base
        base.append(vector_normalizado)
    
    return base

# Funcion que me hace lo de Mpemba1 pero con qutip
def Mpemba1_mejorada_q(L, L_e, autovals, d, N, ini):
  # Hacemos reshape
  L1 = np.reshape(L_e[1], (d.shape[0], d.shape[1]))
  # Diagonalizamos la matriz L1
  todo = (q.Qobj(L1)).eigenstates(sparse = False, sort = 'low')
  vals = todo[0]
  vects = todo[1]

  # Ahora, generamos la base auxiliar para el vector ini
  base_aux = generar_base_ortonormal(ini, N+1)
  base_aux= [q.Qobj(x) for x in base_aux]
  U_cambio = vects[0]*(base_aux[0].dag())
  for i in range(1, N):
      U_cambio += vects[i]*(base_aux[i].dag())

  # Ahora, tenemos que dividir en dos caminos, definimos una tolerancia
  es_cero = [(np.allclose(np.abs(vals[i]), 0.0, atol = 1e-5)) for i in range(len(vals))]

  if(any(es_cero)):
    print('Se puede hacer la vía del cero, con el autovalor: ')
    # Nos vamos al caso en el que un autovalor es 0
    indice = es_cero.index(True)
    U = ini*(vects[indice].dag())
  else:
    print('No se ha podido hacer la via del cero')
    # Si no hay ningun autovalor que sea 0, se coje una pareja de autovalores con signo contrario
    U = np.zeros(d.shape)
  return q.Qobj(U)*q.Qobj(U_cambio), U_cambio

# Funcion que nos calcula la transformacion de Mpemba2, pero con qutip
def Mpemba2_mejorada_q(L, L_e, vals, d, N, ini):
  L_buenos= []
  for elemento in L_e:
      if(np.allclose(L*elemento, np.zeros_like(L*elemento), atol = 1e-2) == False):
          L_buenos.append(elemento)
  # La pasamos a matriz
  L1 = np.reshape(L_buenos[0], (d.shape[0], d.shape[1]))
  L1 = np.reshape(L_e[1], (d.shape[0], d.shape[1]))

  # Diagonalizamos la matriz L1
  todo = (q.Qobj(L1)).eigenstates(sparse = False, sort = 'low')
  autovals = todo[0]
  autovects = todo[1]

  # Ahora, generamos la base auxiliar para el vector estado inicial
  base_aux = generar_base_ortonormal(ini, N+1)
  base_aux = [q.Qobj(x) for x in base_aux]
  
  # Con esto, podemos generar la primera transformacion
  U_cambio = autovects[0]*base_aux[0].dag()
  for i in range(1, N):
      U_cambio += autovects[i]*(base_aux[i].dag())
      
  print('Vamos a probar la via del no cero')
  # Se coje una pareja de autovalores con signo contrario
  i = 0
 
  if(np.real(autovals[0])*np.real(autovals[-1]) < 0):
    F = autovects[0]*autovects[-1].dag() + autovects[-1]*autovects[0].dag()
    s = np.arctan(np.sqrt((np.abs(autovals[0]))/(np.abs(autovals[-1]))))
    print("La s me sale: " + str(s) + ' se ha cogido la s que sale de los autovalores: ')
  
    identidad = q.qeye(F.shape[0])
    U = identidad + (np.cos(s) - 1.0)*(F*F) - 1.j*np.sin(s)*F
  else:
    print('No se puede coger la vía del no cero')
    U = np.zeros(d.shape)
  return q.Qobj(U)*q.Qobj(U_cambio), U_cambio, (autovals[0], autovals[-1])
  
def densidad_p(N, inicial, final):
    j = N/2.0
    n = 2*j + 1
    Jz = q.jmat(j, 'z')
    todo = Jz.eigenstates(sparse = False, sort = 'high')
    base = todo[1]
    
    a, b = inicial + (final-inicial)*random.random(), inicial + (final-inicial)*random.random()
    ini = (a + b*1.j)*base[0]
    for i in range(1, len(base)):
        a, b = inicial + (final-inicial)*random.random(), inicial + (final-inicial)*random.random()
        ini += (a + b*1.j)*base[i]
    
    ini = ini.unit()
    d = ini*ini.dag()
    return d, ini