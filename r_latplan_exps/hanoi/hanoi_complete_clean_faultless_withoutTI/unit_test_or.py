import numpy as np


atoms = [] # = U
for i in range(16):
    atoms.append(f"(z{i})")
for i in range(16):
    atoms.append(f"(not (z{i}))")



ell1 = np.zeros((32)).astype(np.uint8)
ell2 = np.zeros((32)).astype(np.uint8)
ell3 = np.zeros((32)).astype(np.uint8)
ell1[:2] = 1
ell2[0] = 1
ell3[0] = 1
ell3[4] = 1
fake_final_set = []
fake_final_set.append(ell1)
fake_final_set.append(ell2)
fake_final_set.append(ell3)


print(fake_final_set)

atom_groups = []
for row in fake_final_set:
    atom_list = [atoms[i] for i, val in enumerate(row) if val == 1]
    atom_groups.append(atom_list)


or_str = ""

if len(atom_groups) > 0:

    or_str += "( OR "

    for group in atom_groups:

        group_str = ""

        if len(group) == 1:
            group_str += " "+ group[0] +" "

        elif len(group) > 1:
            group_str += " ( AND "

            for ell in group:
                group_str += ell + " "

            group_str += " ) "  
        
        if group_str != "":
            or_str += group_str

    or_str += " )"


print(or_str)