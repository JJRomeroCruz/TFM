#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:33:02 2024

@author: juanjo
"""
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import qutip as q

# Funcion que me obtiene el lindbladiano a partir del hamiltoniano y de los operadores de salto
def limblad(H, J):
    # J tiene que ser una lista
    return q.liouvillian(H, J)