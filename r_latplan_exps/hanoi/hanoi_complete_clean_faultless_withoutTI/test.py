from sympy import symbols
from sympy.logic.boolalg import Or, And, Not, to_cnf

# Original list of lists
clauses = [
    ['(not (z13))', '(not (z5))', '(not (z7))', '(z6)', '(z4)', '(z2)'],
    ['(not (z0))', '(not (z13))', '(z6)', '(z7)', '(z4)', '(z2)'],
    ['(z11)', '(z13)', '(z3)', '(z4)', '(z2)'],
    ['(not (z11))', '(not (z5))', '(z13)', '(z3)', '(z4)', '(z6)', '(z4)', '(z2)'],
    ['(not (z11))', '(not (z6))', '(z13)', '(z3)', '(z4)', '(z2)']
]

# Extract unique variables
variables = set()
for clause in clauses:
    for literal in clause:
        var_name = literal.replace("not ", "").replace(")", "").replace("(", "").strip()
        variables.add(var_name)

# Create symbol mappings
symbol_map = {var: symbols(var) for var in variables}

# Convert clauses to sympy expressions
expression = Or(*[And(*[Not(symbol_map[literal.replace("not ", "").replace(")", "").replace("(", "").strip()])
                         if "(" in literal else symbol_map[literal.strip()]
                         for literal in clause]) for clause in clauses])

# Convert to Conjunctive Normal Form (CNF)
cnf_expression = to_cnf(expression, simplify=True)

print(cnf_expression)
