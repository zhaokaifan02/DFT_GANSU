#!/usr/bin/env python3
from pyscf import gto
import numpy as np

# Build molecule
mol = gto.M(
    atom = '''
    O 0.000000 0.000000 0.117790
    H 0.000000 0.755453 -0.471161
    H 0.000000 -0.755453 -0.471161
    ''',
    basis = 'sto-3g',
    unit = 'angstrom'
)

# Get overlap matrix
S = mol.intor('int1e_ovlp')

print("=== Overlap Matrix ===")
print(S)
print("\n=== Diagonal elements ===")
print(np.diag(S))
