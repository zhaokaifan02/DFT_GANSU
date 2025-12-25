#!/usr/bin/env python3
import numpy as np
from pyscf import gto, dft
import sys

# Build molecule - H2O with coordinates from the code
mol = gto.M(
    atom = '''
    O 0.000000 0.000000 0.117790
    H 0.000000 0.755453 -0.471161
    H 0.000000 -0.755453 -0.471161
    ''',
    basis = 'sto-3g',
    unit = 'angstrom',
    verbose = 0
)

# Setup DFT with LDA
mf = dft.RKS(mol)
mf.xc = 'lda,vwn'  # LDA exchange + VWN correlation
mf.grids.level = 3  # Medium grid (default)

# Run calculation to get converged density
mf.kernel()

print("=== PySCF Debug Information ===\n")

# Nuclear repulsion
print(f"Nuclear repulsion = {mol.energy_nuc()}")

# Grid information
coords = mf.grids.coords
weights = mf.grids.weights
print(f"\nNumber of grid points: {len(coords)}")
print(f"First 10 grid coordinates:")
for i in range(min(10, len(coords))):
    print(f"  g={i}: ({coords[i,0]:12.8f}, {coords[i,1]:12.8f}, {coords[i,2]:12.8f}), w={weights[i]:12.8e}")

# AO values on grid
ao_values = dft.numint.eval_ao(mol, coords)
print(f"\nAO values shape: {ao_values.shape}")
print(f"First 5 grid points, all AOs:")
for i in range(min(5, len(coords))):
    print(f"  g={i}: {ao_values[i,:]}")

# Density on grid
dm = mf.make_rdm1()
rho = dft.numint.eval_rho(mol, ao_values, dm)
print(f"\nDensity on first 10 grid points:")
for i in range(min(10, len(rho))):
    print(f"  g={i}: rho={rho[i]:12.8e}")

# Energy breakdown
h1e = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
j_matrix = mf.get_j(dm=dm)
vxc = mf.get_veff(dm=dm) - j_matrix  # V_eff = J + Vxc, so Vxc = V_eff - J

print(f"\n=== Energy Breakdown ===")
print(f"Total energy = {mf.e_tot}")
print(f"Nuclear repulsion = {mol.energy_nuc()}")
E_core = np.einsum('ij,ji->', dm, h1e)
E_J = 0.5 * np.einsum('ij,ji->', dm, j_matrix)
print(f"Tr(D*H) = {E_core}")
print(f"0.5*Tr(D*J) = {E_J}")

# Compute E_xc directly
ni = dft.numint.NumInt()
exc_value = ni.nr_rks(mol, mf.grids, mf.xc, dm)[1]  # Returns (n_electrons, E_xc, vxc)
print(f"E_xc = {exc_value}")
print(f"Electronic energy = {E_core + E_J + exc_value}")

# Density matrix
print(f"\n=== Density Matrix ===")
print(f"Diagonal elements: {np.diag(dm)}")
print(f"\nFull density matrix:")
print(dm)

# Core Hamiltonian
print(f"\n=== Core Hamiltonian ===")
print(h1e)

# J matrix
print(f"\n=== J Matrix ===")
print(j_matrix)

# Vxc matrix
print(f"\n=== Vxc Matrix ===")
print(vxc)

# Fock matrix
f_matrix = h1e + j_matrix + vxc
print(f"\n=== Fock Matrix (H + J + Vxc) ===")
print(f_matrix)

# Initial guess density
print(f"\n=== Initial Guess Information ===")
mf_init = dft.RKS(mol)
mf_init.xc = 'lda,vwn'
dm_init = mf_init.init_guess_by_minao()
print(f"Initial guess method: minao")
print(f"Initial density matrix diagonal: {np.diag(dm_init)}")
h1e_init = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
e_init = np.einsum('ij,ji->', dm_init, h1e_init)
print(f"Initial Tr(D*H) = {e_init}")
