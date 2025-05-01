from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary

# Define the universal set and subsets
U = {1, 2, 3, 4, 5}
subsets = {
    0: {1, 2, 3},
    1: {2, 4},
    2: {3, 4, 5},
    3: {1, 5}
}

# Define the ILP problem
problem = LpProblem("Set Cover Problem", LpMinimize)

# Define binary variables for each subset
x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in subsets}

# Objective function: Minimize the number of subsets selected
problem += lpSum(x[i] for i in subsets)

# Constraints: Every element in U must be covered by at least one subset
for element in U:
    problem += lpSum(x[i] for i in subsets if element in subsets[i]) >= 1

# Solve the ILP problem
problem.solve()

# Print selected subsets
selected_subsets = [i for i in subsets if x[i].value() == 1]
print("Selected subsets:", selected_subsets)


#### greedy but start with biggest set first

########


########  commence avec les plus petits

##################  si un plus grand implique un plus petit avec même coverage alors élimine le plus petit

##################      



