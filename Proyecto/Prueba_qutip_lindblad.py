#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:11:04 2024

@author: juanjo
"""
import time
import numpy as np
from qutip import tensor, sigmax, sigmay, destroy, identity, spre, spost, liouvillian, Qobj

def dicke_lindbladiano(N, gamma, omega):
    # Definir los operadores
    sx = sigmax()
    sy = sigmay()
    sm = destroy(N)
    id_N = identity(N)
    
    # Construir los operadores del sistema
    Jx = tensor(sx, id_N)
    Jy = tensor(sy, id_N)
    #4Jz = tensor(0.5 * (sm.dag() * sm - id_N), id_N)
    a = tensor(sm, id_N)
    #print(a.shape, Jx.shape, Jy.shape)
    a = sm
    print(a.shape, Jx.shape, Jy.shape)

    # Construir el Hamiltoniano del modelo de Dicke
    H = omega * (Jx + a.dag() + a)
    
    # Construir los términos de Lindblad
    collapse_operators = [np.sqrt(gamma) * a]
    
    # Construir el superoperador de Lindblad
    Lindbladiano = liouvillian(H, collapse_operators)
    
    return Lindbladiano

# Parámetros del modelo de Dicke
inicio = time.time()
N = int(input("Ingrese la dimensión N: "))
gamma = float(input("Ingrese la tasa de decaimiento gamma: "))
omega = float(input("Ingrese la frecuencia omega: "))

#N = int(5)
#gamma = float(3)
#omega = float(4)
# Obtener el Lindbladiano del modelo de Dicke
Lindbladiano_Dicke = dicke_lindbladiano(N, gamma, omega)

print("El Lindbladiano del modelo de Dicke para N =", N, "es:")
print(type(Lindbladiano_Dicke))
fin = time.time()
print('Tiempo: ' + str(fin-inicio))