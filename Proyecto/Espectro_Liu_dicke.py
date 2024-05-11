#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:57:00 2024

@author: juanjo
"""
import numpy as np
import scipy as sc
import sympy as sp
import matplotlib.pyplot as plt
import dicke
import general
#from dicke.py import dicke, densidad
#from general.py import Limblad 

""" En este script vamos a diagonalizar el lindbladiano y ver como es su espectro, lo representamos en el plano complejo """

# Generamos los operadores hamiltonioano y saltos
N = 2
sigma = 1.0
k = 0.1*sigma
g = 1.0*sigma
w = 0.01*sigma
params = [sigma, w, k, g]

H, J = dicke.dicke(N, params)

#Lindbladiano
L, b = general.Limblad(H, [J])

# Lo diagonalizamos
todo = L.eigenvects()
vals = [tup[0] for tup in todo]

re = [sp.re(elemento) for elemento in vals]
im = [sp.im(elemento) for elemento in vals]

plt.xlabel('Parte real')
plt.ylabel('Parte imaginaria')
plt.plot(re, im, 'bo')