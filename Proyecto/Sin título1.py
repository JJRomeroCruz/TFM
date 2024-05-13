#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:27:23 2024

@author: juanjo
"""
import numpy as np
import matplotlib.pyplot as plt
import general
import dicke

# Creamos el lindbladiano del modelo de Dicke
# Generamos el hamiltoniano y los operadores de salto
N = 1
sigma = 1.0
w = 1.0*sigma
k = 8.1*sigma
g = 9.0*sigma
params = [sigma, w, k, g]
H, J = dicke.dicke(N, params)

L, b = general.Limblad(H, [J])
# Sacamos sus autovalores y autovectores


# Obtenemos el valor aplicar el lindbladiano a los autovectores