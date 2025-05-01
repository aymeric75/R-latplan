### TEST IS 
#### REMOVE from preconds of low-lvl actions (in dico_highlvlid_lowlvlactions)
#### ANY GROUP that is in the OR (see atom_groups) or any atom that is in the AND 



import numpy as np


def entails_str(list1, list2):
    return all(elem in list1 for elem in list2)

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

atom_groups = []
for row in fake_final_set:
    atom_list = [atoms[i] for i, val in enumerate(row) if val == 1]
    atom_groups.append(atom_list)


print(atom_groups)

# [['(z0)', '(z1)'], ['(z0)'], ['(z0)', '(z4)']]
# I =  ["(z0)", "(not (z4))"]



dico_highlvlid_lowlvlactions = {}
dico_highlvlid_lowlvlactions[0] = {}
dico_highlvlid_lowlvlactions[0][0] = {}
dico_highlvlid_lowlvlactions[0][1] = {}


dico_highlvlid_lowlvlactions[0][0]["preconds"] = ['(z0)', '(not (z2))', '(not (z12))', '(z1)', '(not (z13))', '(z15)'] 
dico_highlvlid_lowlvlactions[0][1]["preconds"] = ['(z0)', '(not (z8))', '(z4)', '(z10)'] 


dico_highlvlid_lowlvlactions[0][0]["effects"] = ['(not (z0))', '(z12)']
dico_highlvlid_lowlvlactions[0][1]["effects"] = ['(not (z0))', '(z13)']

dico_highlvlid_lowlvlactions_bis = {}
dico_highlvlid_lowlvlactions_bis[0] = dico_highlvlid_lowlvlactions[0]


#

I = ["(z0)", "(not (z4))"]

# print("dico_highlvlid_lowlvlactions_bis[0]")
# print(dico_highlvlid_lowlvlactions_bis[0])

# {0: {'preconds': ['(z0)', '(not (z2))', '(not (z12))', '(not (z13))', '(z15)'], 'effects': ['(not (z0))', '(z12)']}, 
# 1: {'preconds': ['(z0)', '(not (z8))', '(z4)', '(z10)'], 'effects': ['(not (z0))', '(z13)']}}



# for row in final_set:
for lowlvlkey, dico_vals in dico_highlvlid_lowlvlactions[0].items():

    atoms_to_remove = []

    # REMOVING THE AND parts    DONE (works)
    for at in I:

        if at not in atoms_to_remove and at in dico_highlvlid_lowlvlactions[0][lowlvlkey]["preconds"]:
            #print("at is {}".format(at))
            atoms_to_remove.append(at)


    ##### LIST  D ABORD les atoms qui doivent être retirés
    #####

    ############## PUIS à la fin du listing tu les retire tous

    # REMOVING THE OR parts
    # atom_groups are the groups of atoms present in the "general" precondition clause

    
    for gr in atom_groups:

        # if one "general" group is entailed by the low-lvl precondition set THEN
        if entails_str(dico_vals["preconds"], gr):
            

            # we remove from this low level precondition set, all the corresponding atoms
            for ele in gr:
                if ele not in atoms_to_remove:
                    atoms_to_remove.append(ele)                
                # try:
                #     dico_highlvlid_lowlvlactions_bis[0][lowlvlkey]["preconds"].remove(ele)
                # except:
                #     print("...")


    
    for elee in atoms_to_remove:
        dico_highlvlid_lowlvlactions[0][lowlvlkey]["preconds"].remove(elee)


# print("dico_highlvlid_lowlvlactions_bis")
# print(dico_highlvlid_lowlvlactions_bis)
print()
for k, v in dico_highlvlid_lowlvlactions[0].items():

    print(v["preconds"])