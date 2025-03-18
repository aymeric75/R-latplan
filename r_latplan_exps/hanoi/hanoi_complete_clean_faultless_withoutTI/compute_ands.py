import os
import sys
import json
# 
from collections import Counter
from functools import reduce

domainfile="domain.pddl"
problemfile="problem.pddl"

# CODE FOR REMOVING THE NEGATIVE STATES IN THE INIT STATE (AND REMOVING NON PRESENT "EFFECTS" in the goal state)
# Save the current path
#current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the new path (relative to the current file)
translate_path = os.path.join("/workspace/R-latplan", "downward", "src", "translate")

# Temporarily add it to sys.path
sys.path.insert(0, translate_path)


def friendly_effect_name(literal):       
    integer = int(''.join(x for x in str(literal) if x.isdigit()))
    transformed_name = ""
    if "Negated" in str(literal):
        transformed_name += "del_"+str(integer)
    else:
        transformed_name += "add_"+str(integer)
    return transformed_name


# load all lowlvl ids of the high lvl action
num = 3 #15
name_high=""
all_lowLvl_ids = []
# retrieve name

with open('highLvlLowlvlNames.json', 'r') as file:
    data = json.load(file)
    name_high = data[str(num)]


with open('highLvlLowlvl.json', 'r') as file:
    data = json.load(file)
    all_lowLvl_ids = list(data[name_high].keys())

# print("all_lowLvl_ids")
# print(len(all_lowLvl_ids))
# exit()

    
list_of_lists = []

def intersection_of_lists(lists):
    if not lists:
        return []
    
    # Convert the first list to a set and intersect it with the rest
    result_set = set(lists[0])
    for lst in lists[1:]:
        result_set &= set(lst)
    
    return list(result_set)


def count_occurrences(lists):
    # Flatten the list of lists
    flattened_list = [item for sublist in lists for item in sublist]
    
    # Count occurrences using Counter
    occurrence_count = Counter(flattened_list)
    
    return occurrence_count


try:
    #import options  # Replace with actual module name
    import FDgrounder
    from FDgrounder import pddl_parser as pddl_pars

    task = pddl_pars.open(
        domain_filename=domainfile, task_filename=problemfile) # options.task
        

    for trans_id, act in enumerate(task.actions):

        if str(trans_id) in all_lowLvl_ids:

            tmp_act_precond_parts = list(act.precondition.parts)
            tmp_list = []
                
            for precond in tmp_act_precond_parts:
                transformed_name_ = friendly_effect_name(precond)
                tmp_list.append(transformed_name_)

            list_of_lists.append(tmp_list)



    from sympy import symbols
    from sympy.logic.boolalg import SOPform

    

    print(list_of_lists)
    input_data = list_of_lists
    # Step 1: Identify all unique variables
    variables = set()
    for conjunction in input_data:
        for term in conjunction:
            variables.add(term.split('_')[1])

    variables = sorted(list(map(int, variables)))
    num_variables = max(variables) + 1  # To handle all variables from 0 to max

    # Step 2: Convert each conjunction into a binary representation
    def convert_to_binary(conjunction, num_variables):
        binary = ['-' for _ in range(num_variables)]
        for term in conjunction:
            action, var = term.split('_')
            var = int(var)
            binary[var] = '1' if action == 'add' else '0'
        return ''.join(binary)

    binary_terms = [convert_to_binary(conj, num_variables) for conj in input_data]

    # Step 3: Generate minterms for the K-Map
    def expand_dont_care(term):
        if '-' not in term:
            return [term]
        idx = term.index('-')
        with_0 = term[:idx] + '0' + term[idx+1:]
        with_1 = term[:idx] + '1' + term[idx+1:]
        return expand_dont_care(with_0) + expand_dont_care(with_1)

    minterms = []
    for term in binary_terms:
        minterms.extend(expand_dont_care(term))

    minterms = sorted(set(int(mt, 2) for mt in minterms))

    # Step 4: Apply K-Map simplification using SymPy's SOPform
    symbol_list = symbols(f'X0:{num_variables}')
    simplified_expr = SOPform(symbol_list, minterms)

    print(simplified_expr)

    exit()

    # Convert lists to sets for easier set operations
    clause_sets = [set(clause) for clause in list_of_lists]

    # Step 1: Find common elements across all clauses (AND factorization)
    common_elements = set.intersection(*clause_sets)

    # Step 2: Identify remaining unique elements for each clause
    unique_parts = [clause - common_elements for clause in clause_sets]

    # Simplify unique parts by finding common elements among them
    common_in_uniques = set.intersection(*unique_parts)

    # Further reduce unique parts
    reduced_unique_parts = [unique - common_in_uniques for unique in unique_parts]

    # Displaying the results
    print("Common factors across all clauses (can be factored out):")
    print(common_elements)

    print("\nCommon factors in unique parts (further reduction):")
    print(common_in_uniques)

    print("\nReduced unique parts of each clause:")
    for i, unique in enumerate(reduced_unique_parts):
        print(f"Clause {i+1}: {unique}")

    # Final factored form:
    # (common_elements AND common_in_uniques) AND (reduced_unique_part1 OR reduced_unique_part2 OR reduced_unique_part3)

    def format_expression(common, common_unique, uniques):
        combined_common = common.union(common_unique)
        common_expr = " AND ".join(sorted(combined_common))
        unique_exprs = [" AND ".join(sorted(u)) for u in uniques if u]
        disjunction_expr = " OR ".join([f"({expr})" for expr in unique_exprs])
        if disjunction_expr:
            return f"({common_expr}) AND ({disjunction_expr})"
        else:
            return f"({common_expr})"

    factored_expression = format_expression(common_elements, common_in_uniques, reduced_unique_parts)

    print("\nFurther Reduced Boolean Expression:")
    print(factored_expression)

    exit()

    intersec = intersection_of_lists(list_of_lists)
    print("intersecintersec")
    print(intersec)

    occurences = count_occurrences(list_of_lists)
    print(occurences) 
    # 

    # f = open(problemfile, "w")
    # task.domain_name = "latent"
    # f.write(task.get_pddl_problem())
    # f.close()

finally:
    # Restore sys.path to avoid conflicts
    sys.path.pop(0)

