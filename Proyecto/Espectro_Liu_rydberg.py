#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:21:03 2024

@author: juanjo
"""
import numpy as np
import scipy as sc
import sympy as sp
import matplotlib.pyplot as plt
import rydberg
import general
#from dicke.py import dicke, densidad
#from general.py import Limblad 

""" En este script vamos a diagonalizar el lindbladiano y ver como es su espectro, lo representamos en el plano complejo """

# Generamos los operadores hamiltonioano y saltos
N = 1
sigma = 1.0
delta = 0.0*sigma
v = 1.0*sigma
k = 3.0*sigma
params = [sigma, delta, v, k]
H, J = rydberg.rydberg(N, params)

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