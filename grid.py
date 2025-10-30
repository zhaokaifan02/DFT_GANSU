import numpy as np
from pyscf import gto, scf, dft
from scipy.linalg import eigh

# ==== 1. Build integration grid and AO values ====
def load_ao_txt(path):
    with open(path, 'r') as f:
        header = f.readline().strip().split()
        ngrid, nao = map(int, header)
    ao = np.loadtxt(path, skiprows=1)  # shape = (ngrid, nao)
    
    assert ao.shape == (ngrid, nao)
    return ao
def init_grid(mol, grid_add, level=3):
    grids = dft.gen_grid.Grids(mol)
    grids.level = level
    grids.build()
    m_grid = "./work/grids/" + grid_add + "_grid.txt"
    m_ao = "./work/grids/" + grid_add + "_AO.txt"

    data = np.loadtxt(m_grid)
    coords_c = data[:, 0:3]
    weights_c = data[:, 3]
    grids.coords = coords_c
    grids.weights = weights_c
    ao = load_ao_txt(m_ao)



    print("ao shape: ",ao.shape)# 
    print("grid coords shape:", grids.coords.shape) #every x y z no boxes
    return grids, ao

def build(atom_structure, grid_add):
    mol = gto.Mole()
    mol.atom = atom_structure
    mol.basis = 'cc-pvdz'
    mol.cart = True
    mol.spin = None
    mol.build()
    
    nao = mol.nao_nr() 
    nelec = mol.nelec[0] + mol.nelec[1]
    nocc = nelec // 2
    
    print(f"nao: {nao}")
    print(f"nelec: {nelec}")
    print(f"nocc: {nocc}")

    grids, ao_values = init_grid(mol, grid_add)
    print(f"ngrids:  {len(grids.coords)}")
    
    S = mol.intor('int1e_ovlp')
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    Hcore = T + V
    eri = mol.intor('int2e')
    E_nuc = mol.energy_nuc()
    return Hcore, S, nocc, T, eri, ao_values, grids, E_nuc
