import numpy as np
from pyscf import gto, scf, dft
import os
def write_coords_bohr(mol, path="coords_bohr.txt"):
    try:
        coords_bohr = mol.atom_coords(unit='Bohr')
    except TypeError:
        coords_bohr = mol.atom_coords()
    Z = mol.atom_charges().astype(int)
    n = len(Z)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        f.write("Z-coords file (Bohr)\n")
        for z, (x, y, zc) in zip(Z, coords_bohr):
            f.write(f"{z:d} {x:.20e} {y:.20e} {zc:.20e}\n")

    print(f"Wrote {n} atoms to {os.path.abspath(path)}")

if __name__ == "__main__": 
    # ==== 0. Molecule Definition ====
    mol = gto.Mole()
    mol.atom = '''
Br        -0.08890        1.86070        0.13900
Br         1.65600       -0.85330        0.13900
Br        -1.56720       -1.00720        0.13900
C          0.00000       -0.00010       -0.41710
H          0.00000       -0.00010       -1.50970
    ''' 
    mol.basis = 'sto-3g'
    mol.build()
    print(mol.atom_coords())

write_coords_bohr(mol, "Bromoform.txt")