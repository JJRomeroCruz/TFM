#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:30:46 2024

@author: juanjo
"""
import numpy as np
from itertools import combinations

# Definición de las matrices de Pauli
sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Función para generar las matrices de Pauli para un qubit específico
def generate_pauli(alpha, N, k):
    pauli = [np.eye(2)] * N
    pauli[k] = alpha
    result = pauli[0]
    for i in range(1, N):
        result = np.kron(result, pauli[i])
    return result

# Función para calcular el Hamiltoniano del sistema de N qubits
def calcular_hamiltoniano(omega, V, N):
    H = np.zeros((2**N, 2**N), dtype=complex)
    
    # Suma sobre el primer término del Hamiltoniano
    for k in range(N):
        H += omega * generate_pauli(sigma_x, N, k)
    
    # Suma sobre el segundo término del Hamiltoniano
    for h in range(N):
        for k in range(h+1, N):
            H += V * generate_pauli(sigma_z, N, h) @ generate_pauli(sigma_z, N, k)
    
    # Obtenemos el operador de salto
    
    return H

# Parámetros del Hamiltoniano
Omega = 1.0
V = 0.5
N = 4  # Número de qubits

# Calcular el Hamiltoniano
Hamiltoniano = calcular_hamiltoniano(Omega, V, N)
print("Hamiltoniano:")
print(Hamiltoniano)
