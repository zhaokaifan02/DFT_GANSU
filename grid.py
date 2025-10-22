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
    m_grid = "./build/bin/" + grid_add + "_grid.txt"
    m_ao = "./build/bin/" + grid_add + "_SCREEN_AO.txt"

    data = np.loadtxt(m_grid)
    atm_idx_c = data[:, 0].astype(np.int32)
    coords_c = data[:, 1:4]
    weights_c = data[:, 4]
    grids.coords = coords_c
    grids.weights = weights_c
    ao = load_ao_txt(m_ao)
    with open("coords.txt", "w") as f:
        ngrid = coords_c.shape[0]
        f.write(f"{ngrid}\n")
        np.savetxt(f, coords_c, fmt="%.16e")

# 2️⃣ 导出 weights.txt
    with open("weights.txt", "w") as f:
        ngrid = weights_c.shape[0]
        f.write(f"{ngrid}\n")
        np.savetxt(f, weights_c, fmt="%.16e")

# 3️⃣ 导出 AO.txt
    with open("AO.txt", "w") as f:
        ngrid, nao = ao.shape
        f.write(f"{ngrid} {nao}\n")
        np.savetxt(f, ao, fmt="%.16e")
    print("ao shape: ",ao.shape)# 
    print("grid coords shape:", grids.coords.shape) #every x y z no boxes
    return grids, ao

def build(atom_structure, grid_add):
    mol = gto.Mole()
    mol.atom = atom_structure
    mol.basis = 'sto-3g'
    mol.build()
    
    nao = mol.nao_nr() 
    nelec = mol.nelec[0] + mol.nelec[1]
    nocc = nelec // 2
    
    print(f"基函数数: {nao}")
    print(f"电子数: {nelec}")
    print(f"占据轨道数: {nocc}")

    grids, ao_values = init_grid(mol, grid_add)
    print(f"积分网格点数: {len(grids.coords)}")
    
    S = mol.intor('int1e_ovlp')
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    Hcore = T + V
    eri = mol.intor('int2e')
    E_nuc = mol.energy_nuc()
    return Hcore, S, nocc, T, eri, ao_values, grids, E_nuc
