import numpy as np
import csv

# Step 1: Load CSV into a NumPy array
filename = 'FORTEST_final_set_action_8_fined_grained.csv'  # Replace with your actual filename
with open(filename, 'r') as f:
    reader = csv.reader(f)
    data = np.array([[int(cell) for cell in row] for row in reader])

# Step 2: Define atoms
atoms = [f"(z{i})" for i in range(16)] + [f"(not (z{i}))" for i in range(16)]

# Step 3: Associate each row with the corresponding set of atoms
atom_sets = []
nber_of_singletons = 0
for row in data:
    active_atoms = [atoms[i] for i, val in enumerate(row) if val == 1]
    atom_sets.append(active_atoms)
    if len(active_atoms) == 1:
        nber_of_singletons += 1




# Example: Print the result
for i, atom_set in enumerate(atom_sets):
    print(f"Row {i}: {atom_set}")

print("nber_of_singletons is {}".format(str(nber_of_singletons)))