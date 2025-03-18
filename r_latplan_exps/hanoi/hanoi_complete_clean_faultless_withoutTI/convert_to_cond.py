import os
import sys
import numpy as np
from sympy import symbols
from sympy.logic.boolalg import Or, And, Not, simplify_logic




domainfile = "domainNoDupp.pddl"
problemfile = "pbs_normalR-Latplan-N16/1_0/ama3_r_latplan_exps_hanoi_hanoi_complete_clean_faultless_withoutTI_domain_blind_problem.pddl"




def simplify_clauses(clauses):

    # Extract all unique variable names
    variables = set()
    for clause in clauses:
        for expr in clause:
            var = expr.replace("(not ", "").replace(")", "").replace("(", "").strip()
            variables.add(var)

    # Define SymPy symbols
    symbols_dict = {var: symbols(var) for var in variables}

    #print(symbols_dict)

    # Convert text to SymPy expressions
    def parse_expression(expr):
        expr = expr.strip()
        if expr.startswith("(not "):  # Negation case
            var = expr.replace("(not ", "").replace(")", "").replace("(", "").strip()
            return Not(symbols_dict[var])
        else:
            return symbols_dict[expr.replace("(", "").replace(")", "").replace("(", "").strip()]

    # Convert list of clauses to SymPy logic expressions
    sympy_clauses = [And(*[parse_expression(e) for e in clause]) for clause in clauses]

    # Combine clauses with OR
    full_expr = Or(*sympy_clauses)

    # Simplify the expression
    simplified_expr = simplify_logic(full_expr, form='dnf')  # Disjunctive Normal Form

    # print("Simplified Expression:")
    # print(simplified_expr)
    num_literals = sum(1 for arg in simplified_expr.args for _ in (arg.args if isinstance(arg, And) else [arg]))
    #print(num_literals)
    num_disjunctions = len(simplified_expr.args) if isinstance(simplified_expr, Or) else 1
    print("num disjunctions ")
    print(num_disjunctions)
    return


def load_dataset(path_to_file):
    import pickle
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    # print("path_to_file path_to_file")
    # print(path_to_file)
    # exit()
    return loaded_data



current_dir = os.path.dirname(os.path.abspath(__file__))
translate_path = os.path.join(current_dir, "downward", "src", "translate")
sys.path.insert(0, translate_path)

def friendly_name(literal):       
    integer = int(''.join(x for x in str(literal) if x.isdigit()))
    transformed_name = ""
    if "Negated" in str(literal):
        transformed_name += "(not (z"+str(integer)+"))"
    else:
        transformed_name += "(z"+str(integer)+")"
    return transformed_name
two_tabs_space  = "         "
def return_whens(dico_low_level_actions, low_ids):

    retour = ""

    for lid in low_ids:

        tmp = two_tabs_space+"(when "

        # Writting preconds
        dico_preconds = dico_low_level_actions[lid]["preconds"]
        if len(dico_preconds) > 1:
            tmp += " ( and "
            for precond in dico_preconds:
                tmp += precond + " "
            tmp += ")\n"
        
        else:
            for precond in dico_preconds:
                tmp += precond
            tmp += ")\n"


        # Writting effects
        dico_effects = dico_low_level_actions[lid]["effects"]
        tmp += two_tabs_space
        if len(dico_effects) > 1:
            tmp += " ( and "
            for eff in dico_effects:
                tmp += eff + " "
            tmp += "\n"+two_tabs_space+")"+"\n"

        else:
            for eff in dico_effects:
                tmp += eff
            tmp += "\n"+two_tabs_space+")"+"\n"

        tmp += two_tabs_space+")"
        

        retour += "\n"+tmp
    
    return retour
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

    # like 1253 : {
    #   'effects': [(z1)]
    #   'preconds': [(not (z1)), (z5)]
    #}

    # for each low lvl action, you retrieve the set of preconditions and of effects
    # you write each precond like (z3) or (not (z3))
    # each effect like the same
    

    # 1459 instead of 1469

    # task.indices_actions_no_effects


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



    for high_id in range(22):

        low_ids = dico_transitions_per_high_lvl_actions[high_id]

        print("low_idslow_idslow_ids")
        print(low_ids)

        add_effs = [f"(z{i})" for i in range(16)]
        del_effs = [f"(not (z{i}))" for i in range(16)]

        groups_of_preconds_per_effects = {}
        for add_eff in add_effs:
            groups_of_preconds_per_effects[add_eff] = []
            for lid in low_ids:
                if len(dico_low_level_actions[lid]['preconds']) > 0:
                    groups_of_preconds_per_effects[add_eff].append(dico_low_level_actions[lid]['preconds'])
        
        for del_eff in del_effs:
            groups_of_preconds_per_effects[del_eff] = []
            for lid in low_ids:
                if len(dico_low_level_actions[lid]['preconds']) > 0:
                    groups_of_preconds_per_effects[del_eff].append(dico_low_level_actions[lid]['preconds'])
        
        print(groups_of_preconds_per_effects["(z1)"])

        print(len(groups_of_preconds_per_effects["(not (z1))"]))

        #print(len(groups_of_preconds_per_effects))

        simplify_clauses(groups_of_preconds_per_effects["(not (z1))"])
        exit()
        # group (disjunction) of groups (conjunctions)

    # per effect, factorize the preconds groups

    # factorize the effects ? 

    # i.e. look at preconds group that are the same

    #           ===> then you can make a conditional effect with two effects 

    # 1) for each high lvl action, for each effect, retrieve all the precond groups (as OR formulas)


    # Starting writing


    with open("domainCondProsaic.pddl", "w") as f:


        f.write("(define (domain latent)\n")
        f.write("(:requirements :negative-preconditions :strips :conditional-effects)\n")
        f.write("(:types\n")
        f.write(")\n")
        f.write("(:predicates\n")

        for i in range(16):
            f.write("(z"+str(i)+" )\n")
        f.write(")\n")

        for high_id in range(22):

            low_ids = dico_transitions_per_high_lvl_actions[high_id]

            f.write("(:action a"+str(high_id)+"\n")
            f.write("   :parameters ()\n")
            f.write("   :precondition ()\n")
            f.write("   :effect (and\n")


            thestring = return_whens(dico_low_level_actions, low_ids)

            print(thestring)
            f.write("   "+thestring)



    # ############ CODE FOR REMOVING NON PRESENT "EFFECTS" in the goal states ##################
    # effectsToKeep = []
    # for trans_id, act in enumerate(task.actions):
    #     tmp_act_effects = act.effects

    #     for eff in tmp_act_effects:
    #         # print("EFF LITERAL")
    #         # print(eff.literal)
    #         integer = int(''.join(x for x in str(eff.literal) if x.isdigit()))
    #         transformed_name = ""
    #         if "Negated" in str(eff.literal):
    #             transformed_name += "del_"+str(integer)
    #         else:
    #             transformed_name += "add_"+str(integer)

    #         if transformed_name not in effectsToKeep:
    #             effectsToKeep.append(transformed_name)

    # goal_parts = list(task.goal.parts)

    # for part in goal_parts:
    #     integer = int(''.join(x for x in str(part) if x.isdigit()))
    #     transformed_name = ""
    #     if "Negated" in str(part):
    #         transformed_name += "del_"+str(integer)
    #     else:
    #         transformed_name += "add_"+str(integer)

    #     print("transformed_name is {}".format(transformed_name))

    #     if transformed_name not in effectsToKeep:
    #         goal_parts.remove(part)
    # task.goal.parts = tuple(goal_parts)


    # ############ END CODE FOR REMOVING NON PRESENT "EFFECTS" in the goal states ##################


    # f = open(problemfile, "w")
    # task.domain_name = "latent"
    # f.write(task.get_pddl_problem())
    # f.close()

finally:
    # Restore sys.path to avoid conflicts
    sys.path.pop(0)
