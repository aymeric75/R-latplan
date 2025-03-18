import os
import sys
import numpy as np
from sympy import symbols
from sympy.logic.boolalg import Or, And, Not, simplify_logic
from itertools import combinations

domainfile = "domainNoDupp.pddl"
problemfile = "pbs_normalR-Latplan-N16/1_0/ama3_r_latplan_exps_hanoi_hanoi_complete_clean_faultless_withoutTI_domain_blind_problem.pddl"


def load_dataset(path_to_file):
    import pickle
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)

    return loaded_data

def friendly_name(literal):       
    integer = int(''.join(x for x in str(literal) if x.isdigit()))
    transformed_name = ""
    if "Negated" in str(literal):
        transformed_name += "(not (z"+str(integer)+"))"
    else:
        transformed_name += "(z"+str(integer)+")"
    return transformed_name


def create_rules(preconds, effects):
    from itertools import product
    pairs = list(product(preconds, effects))
    return pairs


def detect_contradictions(rules):
    """
    Detect contradictions in a set of IF-THEN rules.
    
    :param rules: A list of tuples (condition, effect)
    :return: List of contradictions found
    """
    contradictions = []
    rule_dict = {}

    # Build rule dictionary where conditions map to sets of effects
    for condition, effect in rules:
        if condition not in rule_dict:
            rule_dict[condition] = set()
        rule_dict[condition].add(effect)

    # Check for contradictions within the same condition
    for condition, effects in rule_dict.items():
        for eff1, eff2 in combinations(effects, 2):
            if effects_are_conflicting(eff1, eff2):
                contradictions.append(f"Contradiction: IF {condition} THEN {eff1} vs IF {condition} THEN {eff2}")

    # Check for contradictions between mutually exclusive conditions
    conditions = list(rule_dict.keys())
    for cond1, cond2 in combinations(conditions, 2):
        if conditions_are_mutually_exclusive(cond1, cond2):
            common_effects = rule_dict[cond1] & rule_dict[cond2]
            if common_effects:
                contradictions.append(f"Inconsistency: {cond1} and {cond2} cannot be true together, yet they both imply {common_effects}")

    return contradictions

def effects_are_conflicting(eff1, eff2):
    """
    Checks if two effects are contradictory.
    
    :param eff1: Effect 1
    :param eff2: Effect 2
    :return: Boolean indicating if they conflict
    """
    return (f"(not {eff1})" == eff2) or (f"(not {eff2})" == eff1)

def conditions_are_mutually_exclusive(cond1, cond2):
    """
    Checks if two conditions are mutually exclusive.
    
    :param cond1: Condition 1
    :param cond2: Condition 2
    :return: Boolean indicating if they are mutually exclusive
    """
    return (f"(not {cond1})" == cond2) or (f"(not {cond2})" == cond1)





def parse_literal(literal):
    """Parses a literal, returning (variable, is_positive)."""
    if literal.startswith('(not '):
        return literal[6:-2], False
    return literal[1:-1], True

def detect_contradictions_in_pairs(pairs):
    """Detects logical contradictions in a list of (preconditions, effects) pairs."""
    contradictions = []
    
    for i, (preconds, effects) in enumerate(pairs):
        precond_set = {parse_literal(lit) for lit in preconds}
        effect_set = {parse_literal(lit) for lit in effects}
        
        # contradictions WITHIN
        # == if 

        # contradictions BETWEEN


        # # Self-contradictions: an effect negates a precondition
        # for var, val in effect_set:
        #     if (var, not val) in precond_set:
        #         contradictions.append((f"Self-contradiction in pair {i}: {var}"))
        
        # print("contradictions")
        # print("precond_set")
        # print(precond_set)
        # print("effect_set")
        # print(effect_set)
        # print(contradictions)
        # exit()

        # Cross-pair contradictions
        for j, (other_preconds, other_effects) in enumerate(pairs):
            if i >= j:  # Avoid duplicate comparisons
                continue
            
            other_precond_set = {parse_literal(lit) for lit in other_preconds}
            other_effect_set = {parse_literal(lit) for lit in other_effects}
            
            # Contradiction: an effect of one pair negates a precondition of another
            for var, val in effect_set:
                if (var, not val) in other_precond_set:
                    contradictions.append((f"Contradiction between pair {i} and {j}: {var}"))
            
            # Contradiction: an effect of one pair negates an effect of another
            for var, val in effect_set:
                if (var, not val) in other_effect_set:
                    contradictions.append((f"Effect contradiction between pair {i} and {j}: {var}"))
    
    return contradictions







def count_preconds_and_effects(low_ids, dico_low_level_actions):

    # Counting the effects

    add_effs = [f"(z{i})" for i in range(16)]
    del_effs = [f"(not (z{i}))" for i in range(16)]

    pos_preconds = [f"(z{i})" for i in range(16)]
    neg_preconds = [f"(not (z{i}))" for i in range(16)]

    dico_effects = {}
    for add in add_effs:
        dico_effects[add] = 0
    for dell in del_effs:
        dico_effects[dell] = 0
    for lid in low_ids:
        for k in list(dico_effects.keys()):
            if k in dico_low_level_actions[lid]["effects"]:
                dico_effects[k] += 1

    # Counting the preconds

    dico_preconds = {}
    for pos in pos_preconds:
        dico_preconds[pos] = 0
    for neg in neg_preconds:
        dico_preconds[neg] = 0
    for lid in low_ids:
        for k in list(dico_preconds.keys()):
            if k in dico_low_level_actions[lid]["preconds"]:
                dico_preconds[k] += 1

    print("dico preconds")
    print(dico_preconds)

    #dico_low_level_actions[lid]["preconds"]

    return dico_preconds, dico_effects

try:
    #import options  # Replace with actual module name
    import FDgrounder
    from FDgrounder import pddl_parser as pddl_pars

    task = pddl_pars.open(
        domain_filename=domainfile, task_filename=problemfile) # options.task
        

    ##### 0) retrieve dico of high lvl VS low level

    dico_transitions_per_high_lvl_actions = {}
    path_to_dataset = "/workspace/R-latplan/r_latplan_datasets/hanoi/hanoi_complete_clean_faultless_withoutTI/data.p"
    # load dataset for the specific experiment
    loaded_data = load_dataset(path_to_dataset)
    train_set_no_dupp = loaded_data["train_set_no_dupp_processed"]
    all_high_lvl_actions_unique = loaded_data["all_high_lvl_actions_unique"]

    for ii, ele in enumerate(train_set_no_dupp):
        if int(np.argmax(ele[2])) not in dico_transitions_per_high_lvl_actions:
            dico_transitions_per_high_lvl_actions[int(np.argmax(ele[2]))] = []
        if int(np.argmax(ele[1])) not in dico_transitions_per_high_lvl_actions[int(np.argmax(ele[2]))]:
            dico_transitions_per_high_lvl_actions[int(np.argmax(ele[2]))].append(int(np.argmax(ele[1])))


    ##### 1) convert R-Latplan to conditional PDDL V1

    dico_low_level_actions = {}

    nb_total_actions = len(task.actions) + len(task.indices_actions_no_effects) # including actions with no effects
    cc_normal_actions = 0

    for trans_id in range(nb_total_actions):

        dico_low_level_actions[trans_id] = {}

        if trans_id in task.indices_actions_no_effects:
            dico_low_level_actions[trans_id]['effects'] = []
            dico_low_level_actions[trans_id]['preconds'] = []
        else:

            act = task.actions[cc_normal_actions]
            
            dico_low_level_actions[trans_id]['effects'] = []
            tmp_act_effects = act.effects
            for eff in tmp_act_effects:
                transformed_name = friendly_name(eff.literal)
                dico_low_level_actions[trans_id]['effects'].append(transformed_name)

            dico_low_level_actions[trans_id]['preconds'] = []
            tmp_act_precond_parts = list(act.precondition.parts)
            for precond in tmp_act_precond_parts:
                transformed_name_ = friendly_name(precond)
                dico_low_level_actions[trans_id]['preconds'].append(transformed_name_)

            cc_normal_actions += 1



    #### DATA PROCESSING 
    # NOW THAT WE HAVE THE DATA, WE CREATE A DIR

    # dir_name = "STATS-R-LATPLAN"
    # os.makedirs(dir_name, exist_ok=True)

    # for highLvl_id in range(22):
            
        
    #     file_name = "a"+str(highLvl_id)
    #     file_path = os.path.join(dir_name, file_name)
    #     # Write to the file
    #     with open(file_path, "w") as file:

    #         # Preconds Occurences
    #         # dico_low_level_actions & dico_transitions_per_high_lvl_actions
            
    #         low_ids = dico_transitions_per_high_lvl_actions[highLvl_id]

    #         preconds, effects = count_preconds_and_effects(low_ids, dico_low_level_actions)

    #         file.write("#transitions: "+str(len(low_ids))+"\n")

    #         file.write("Preconds stats:\n")
    #         for k, v in preconds.items():
    #             file.write(str(k)+": "+str(v)+"\n")

    #         file.write("Effects stats:\n")
    #         for k, v in effects.items():
    #             file.write(str(k)+": "+str(v)+"\n")
    list_of_pairs = []
    # IDENTIFYING inconsistencies AND contraditions among IF precond THEN effects rules
    for highLvl_id in range(22):

        # dico_low_level_actions & dico_transitions_per_high_lvl_actions
        
        low_ids = dico_transitions_per_high_lvl_actions[highLvl_id]

        list_of_if_then_for_high_lvl_action = []

        #list_of_pairs = []

        for lid in low_ids:
            
            preconds = dico_low_level_actions[lid]["preconds"]
            effects = dico_low_level_actions[lid]["effects"]
            # print("preconds")
            # print(preconds)
            # print("effects")
            # print(effects)
            tmp_tuple = tuple([preconds, effects])
            list_of_pairs.append(tmp_tuple)

            tmp_list = create_rules(preconds, effects)
            list_of_if_then_for_high_lvl_action.extend(tmp_list)

        print("list_of_if_then_for_high_lvl_action")
        print(list_of_if_then_for_high_lvl_action)

        
        # if contradictions:
        #     print("Contradictions found:")
        #     for c in contradictions:
        #         print(c)
        # else:
        #     print("No contradictions found.")

        # Run contradiction detection
        contradictions = detect_contradictions_in_pairs(list_of_pairs)

        print(len(contradictions))
        exit()

# you are right, also I realized that it could even not be feasible to factorize anything since there are contradictions among the 
#
#  if then rules between preconditions and effects.  I made a small algorithm that - for a same high lvl action - 1) finds out all the IF
#
#
# one precondition  THEN one effects, and already there 



finally:
    # Restore sys.path to avoid conflicts
    sys.path.pop(0)
