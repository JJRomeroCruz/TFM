#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:52:28 2024

@author: juanjo
"""
from qutip import Qobj
import qutip as q
import numpy as np

def base_ortonormal_con_vector(vector):
    # Convertir el vector a un objeto Qobj
    vector_qobj = Qobj(vector)

    # Normalizar el vector
    vector_normalizado = vector_qobj.unit()

    # Definir el primer vector de la base como el vector normalizado
    base = [vector_normalizado]

    # Generar los vectores restantes de la base utilizando Gram-Schmidt
    for i in range(len(vector)):
        if i == 0:
            continue
        # Proyectar el vector original sobre los vectores de la base ya generados
        proyecciones = [vector_normalizado.overlap(v) * v for v in base]
        # Restar las proyecciones del vector original para ortogonalizarlo
        vector_ortogonalizado = vector_qobj - sum(proyecciones)
        # Normalizar el vector ortogonalizado para obtener un nuevo vector de la base
        vector_normalizado = vector_ortogonalizado.unit()
        # Agregar el nuevo vector a la base
        base.append(vector_normalizado)

    return base

# Ejemplo de uso:
# Definir el vector incluido en la base
mi_vector = np.array([1, 2, 3])

# Generar la base ortonormal que contiene el vector
mi_base = base_ortonormal_con_vector(mi_vector)

# Imprimir la base generada
print("Base ortonormal que contiene el vector:")
for vector in mi_base:
    print(vector.full())

q.about()