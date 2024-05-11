#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:09:49 2024

@author: juanjo
"""
import matplotlib.pyplot as plt
import numpy as np
import qutip as q

dim = 100
jx = q.jmat((dim -1)/2.0, 'x')
jy = q.jmat((dim -1)/2.0, 'y')
jz = q.jmat((dim -1)/2.0, 'z')

e_ops = [jx, jy, jz]

H = (jz + jx)/2

psi0 = q.rand_ket(dim, seed = 100393394)
tlist = np.linspace(0.0, 10.0, 200)

#results = q.krylovsolve(H, psi0, tlist, krylov_dim = 100, e_ops = e_ops, options = {store_states: True, progress_bar: 'text'})
results = q.krylovsolve(H, psi0, tlist, 100, e_ops, q.Options(store_states = True))

plt.figure()
for expect in results.expect:
    plt.plot(tlist, expect)
plt.legend(('jmat x', 'jmat y', 'jmat z'))
plt.xlabel('Time')
plt.ylabel('Expectation values')
plt.show()
print(type(results.states))