#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:37:08 2024

@author: juanjo
"""
import numpy as np
import ppm
import matplotlib.pyplot as plt
import networkx as nx

""" En este script se obtiene la representacion de la red trófica para varios valores de T """

# Establecemos los parametros del modelo
B = 40
N = 77
L = 181
T = 1

# Generamos la red trofica
G = ppm.ppm(B, N, L, T)

# Representamos la red
ppm.dibujar(G)