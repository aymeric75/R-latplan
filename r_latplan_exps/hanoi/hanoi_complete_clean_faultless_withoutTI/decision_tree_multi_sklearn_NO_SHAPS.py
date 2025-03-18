import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz, plot_tree
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.tree import _tree
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score
import json
import os
import sys
from collections import defaultdict
import math
from functools import reduce
import argparse
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)




###############################################################
#############            Hyper params               ###########
###############################################################

base_dir = "/workspace/R-latplan/r_latplan_exps/hanoi/hanoi_complete_clean_faultless_withoutTI/"


parser = argparse.ArgumentParser(description="Process and create a directory based on input parameters.")

# Boolean arguments
parser.add_argument("--filter_out_dt1", choices=["true", "false"], required=True)
parser.add_argument("--factorize_dt1", choices=["true", "false"], required=True)
parser.add_argument("--filter_out_dt2", choices=["true", "false"], required=True)

# Exclusive group (only one can be true)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--x_all_atoms_dt2", action="store_true")
group.add_argument("--x_atoms_from_cond_effects_dt2", action="store_true")
group.add_argument("--x_atoms_outisde_of_cond_effects_dt2", action="store_true")

parser.add_argument("--index", required=True)

# # Algorithm type (one of four values)
# parser.add_argument("--algorithm", choices=["blind", "lama", "lmcut", "mands"], required=True)

args = parser.parse_args()


# at least one of the 3 below must be true
x_all_atoms_dt2 = False
x_atoms_from_cond_effects_dt2 = False
x_atoms_outisde_of_cond_effects_dt2 = False

# Determine which exclusive option is true
if args.x_all_atoms_dt2:
    x_all_atoms_dt2 = True
    exclusive_param = "x_all_atoms_dt2"
elif args.x_atoms_from_cond_effects_dt2:
    x_atoms_from_cond_effects_dt2 = True
    exclusive_param = "x_atoms_from_cond_effects_dt2"
elif args.x_atoms_outisde_of_cond_effects_dt2:
    x_atoms_outisde_of_cond_effects_dt2 = True
    exclusive_param = "x_atoms_outisde_of_cond_effects_dt2"

assert x_all_atoms_dt2 ^ x_atoms_from_cond_effects_dt2 ^ x_atoms_outisde_of_cond_effects_dt2



# Construct directory name
dir_name = f"{args.index}__{args.filter_out_dt1}__{args.factorize_dt1}__{args.filter_out_dt2}__{exclusive_param}"





# Create the directory
# os.makedirs(dir_name, exist_ok=True)
# print(f"Directory created: {dir_name}")


filter_out_dt1 = True if args.filter_out_dt1 == "true" else False # remove the branches (i.e. the conditional effects) that have less samples than the mean (of samples per branch in the tree)
factorize_dt1 = True if args.factorize_dt1 == "true" else False  # if cond effects with same effect group, factorize their preconds
filter_out_dt2 = True if args.filter_out_dt2 == "true" else False  # same as filter_out_dt1 but for dt2

print("{} {} {} {}".format(filter_out_dt1, factorize_dt1, filter_out_dt2, exclusive_param))


#### coverage instead of the nber of samples that a branch covers

####### ok, pense à une precond (AND (z0) (not (z3)))

###################   si le nbre de samples c'est 5,

################### et pour une autre le nbre de samples c'est 1

################### MAIS si le sample covered par la 2e est présent dans le set de la 1iere ALORS la coverage du total reste 5 !!!

#### par "ele"  (ie a precond or an effect) tu veux les exacts IDs des transitions qu'elle couvre

##### ENSUITE: lors du choix des effects par exemple: tu prends, par exemple, les effects suffisant pour avoir le max de coverage

#################       ===> tu pars des effets qui cvre le plkus grand nbre de samples Et tu descends !!!


########  
### a bash script that will 
### 1)
### 
### situation de base
####      tout False SAUF   x_atoms_outisde_of_cond_effects_dt2
####      
####       ensuite, unit tests:
####
####         1)   tout à False et test x_all_atoms_dt2, x_atoms_from_cond_effects_dt2,              OK
####         2)   test x_all_atoms_dt2, x_atoms_from_cond_effects_dt2 à False et test un par un    
####
####            filter_out_dt1, factorize_dt1, filter_out_dt2 et only_supersets_dt2
####
####                    filter_out_dt1 OK

####                    

#### bash that 1) clan pbs folders 2) call decision_tree_ stuff and create 





###############################################################
####################### UTILS FUNCTIONS and CLASSES ###########
###############################################################

def dico_disjunctions_per_high_lvl_action():

    # 1à) retrieve 

    dico = {}

    for num in range(22):

        name_high=""
        all_lowLvl_ids = []
        with open('highLvlLowlvlNames.json', 'r') as file:
            data = json.load(file)
            name_high = data[str(num)]
        
        with open('highLvlLowlvl.json', 'r') as file:
            data = json.load(file)
            all_lowLvl_ids = list(data[name_high].keys())

        domainfile="domain.pddl"
        problemfile="problem.pddl"
        translate_path = os.path.join("/workspace/R-latplan", "downward", "src", "translate")
        sys.path.insert(0, translate_path)

        try:
            import FDgrounder
            from FDgrounder import pddl_parser as pddl_pars
            task = pddl_pars.open(
                domain_filename=domainfile, task_filename=problemfile) # options.task
            
            list_of_lists = []
            for trans_id, act in enumerate(task.actions):

                if str(trans_id) in all_lowLvl_ids:

                    tmp_act_precond_parts = list(act.precondition.parts)
                    tmp_list = []
                        
                    for precond in tmp_act_precond_parts:
                        transformed_name_ = friendly_effect_name(precond)

                        tmp_list.append(transformed_name_)

                    list_of_lists.append(tmp_list)

            dico[num] = list_of_lists

        finally:
            # Restore sys.path to avoid conflicts
            sys.path.pop(0)

    return dico


def dico_disjunctions_simple_per_high_lvl_action():

    # 1à) retrieve 

    dico = {}

    for num in range(22):

        name_high=""
        all_lowLvl_ids = []
        with open('highLvlLowlvlNames.json', 'r') as file:
            data = json.load(file)
            name_high = data[str(num)]
        
        with open('highLvlLowlvl.json', 'r') as file:
            data = json.load(file)
            all_lowLvl_ids = list(data[name_high].keys())

        domainfile="domain.pddl"
        problemfile="problem.pddl"
        translate_path = os.path.join("/workspace/R-latplan", "downward", "src", "translate")
        sys.path.insert(0, translate_path)

        try:
            import FDgrounder
            from FDgrounder import pddl_parser as pddl_pars
            task = pddl_pars.open(
                domain_filename=domainfile, task_filename=problemfile) # options.task
            
            list_of_lists = []
            for trans_id, act in enumerate(task.actions):

                if str(trans_id) in all_lowLvl_ids:

                    tmp_act_precond_parts = list(act.precondition.parts)
                    tmp_list = []
                        
                    for precond in tmp_act_precond_parts:
                        transformed_name_ = friendly_effect_name(precond)

                        tmp_list.append(transformed_name_)

                    list_of_lists.extend(tmp_list)

            dico[num] = [list(set(list_of_lists))]

        finally:
            # Restore sys.path to avoid conflicts
            sys.path.pop(0)

    return dico


def format_precond(precond, reverse=False):
    splits = precond.split('_')
    if reverse:
        if int(precond[-1]) == 0:
            return '('+str(splits[0])+str(splits[1])+')'
        elif int(precond[-1]) == 1:
            return '(not ('+str(splits[0])+str(splits[1])+'))'
    else:
        if int(precond[-1]) == 0:
            return '(not ('+str(splits[0])+str(splits[1])+'))'
        elif int(precond[-1]) == 1:
            return '('+str(splits[0])+str(splits[1])+')'


# return the opposite
# eg if input is (z0) then returns (not (z0))
def opposite(atom):
    integer = str(int(''.join(x for x in str(atom) if x.isdigit())))
    if "not" in atom:
        return "(z"+integer+")"
    else:
        return "(not (z"+integer+"))"

# precond like (not (z10))
def reverse_format(precond):
    retour = "z_"
    integer = int(''.join(x for x in str(precond) if x.isdigit()))
    retour += str(integer) + "_"
    if "not" in precond:
        retour += "0"
    else:
        retour += "1"
    return retour


def load_dataset(path_to_file):
    import pickle
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data


def friendly_effect_name(literal):       
    integer = int(''.join(x for x in str(literal) if x.isdigit()))
    transformed_name = ""
    if "Negated" in str(literal):
        transformed_name += "del_"+str(integer)
    else:
        transformed_name += "add_"+str(integer)
    return transformed_name


def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            sample_count = tree_.n_node_samples[node]  # Get the number of samples at this leaf
            class_label = np.argmax(tree_.value[node])
            path += [f"class is: {class_label}, samples: {sample_count}"]
            paths += [path]
            
    recurse(0, path, paths)
    return paths


class UnionFind:
    def __init__(self):
        self.parent = {}
    
    def find(self, x):
        # Path compression
        if x not in self.parent:
            self.parent[x] = x
            return x
        elif self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)


def factorize_dict(input_dict, num_action=None):



    effects_for_all = []
    for keyy, listes in input_dict.items():
        if len(listes) == 1:
            if len(listes[0]) == 1:
                if isinstance(listes[0][0], int):
                    effects_for_all.append(keyy)



    list_to_keys = defaultdict(set)
    new_dict = defaultdict(list)
    
    # Reverse mapping from lists to keys
    for key, lists in input_dict.items():
        # if not lists:
        #     new_dict[(key,)] = []  # Preserve empty lists with single key as tuple

        if key not in effects_for_all:
            for lst in lists:
                tuple_lst = tuple(sorted(lst[:-1]))  # Convert list to tuple for immutability and sorting for consistency
                # print("tuple_lsttuple_lsttuple_lst")
                # print(tuple_lst)
                # print(key)
                # exit()
                list_to_keys[tuple_lst].add(key)
        
                for sup in effects_for_all:
                    list_to_keys[tuple_lst].add(sup)

    # Construct the new factorized dictionary
    processed_keys = set()
    # pour chaque combi de preconds (lst) abd corresponding effects (keys)
    #
    for lst, keys in list_to_keys.items():
        keys_tuple = tuple(sorted(keys))
        new_dict[keys_tuple].append(list(lst))
        processed_keys.update(keys)
    
    # Convert keys tuples to appropriate format
    factorized_dict = {}
    for keys, lists in new_dict.items():
        factorized_dict[tuple(keys)] = lists  # Ensure all keys are stored as tuples
    

    # print("factorized_dictfactorized_dictfactorized_dict")
    # print(factorized_dict)
    # exit()
    return factorized_dict


# list like [['(z11)', '(z13)', '(z3)'], ['(not (z11))', '(not (z6))', '(z13)', '(z3)']]
def logic_factorization(list_of_clauses):

    # Convert lists to sets for easier set operations
    clause_sets = [set(clause) for clause in list_of_clauses]

    # print()
    # print("clause_sets")
    # print(clause_sets)

    # [
    #     {'(z3)', '(z13)', '(z11)'}, 
    #     {'(z3)', '(z13)', '(not (z6))', '(not (z11))'}
    # ]
    common_elements_str = ""
    not_common_elements_str = ""
    # Step 1: Find common elements across all clauses (AND factorization)
    common_elements = set.intersection(*clause_sets)

    #### LE TRAITEMENT CI DESSOUS TU LE FAIS A LA FIN
    # si common elements ==> (AND comon_eles not_comon_eles)   !!!!!!!!!!!!!!
    # sinon             ===> not_comon_eles
    
    # POUR L INSTANT CALCUL SEPARE DE chacun (common_elements et not_comon_eles)
    if len(common_elements) > 0:
        if len(common_elements) == 1:
            common_elements_str += list(common_elements)[0]

        else:
            common_elements_str += "(AND "
            for ele in common_elements:
                common_elements_str += ele + " "
            common_elements_str += ")"
    # print()
    # print("common_elements")
    # print(common_elements)
    # print()
    # print("common_elements_str")
    # print(common_elements_str)

    # Step 2: Identify remaining unique elements for each clause
    unique_parts = [clause - common_elements for clause in clause_sets]




    # (AND common_eles (OR ele1diff ele2diff2))
    if len(unique_parts) > 0:
        if len(unique_parts) == 1:

            not_common_elements_str += list(unique_parts[0])[0]

        else:
            
            # 
            not_common_elements_str += " (OR "


            for ele in list(unique_parts):
                
                if len(ele) == 1:
                    not_common_elements_str += list(ele)[0] + " "
                else:
                    not_common_elements_str += "(AND "
                    for e in list(ele):
                        not_common_elements_str += e + " "
                    not_common_elements_str += ")"



            not_common_elements_str += ")"
    
    # print()
    # print("not_common_elements_str")
    # print(not_common_elements_str)
    # exit()

    
    #### LE TRAITEMENT CI DESSOUS TU LE FAIS A LA FIN
    # si common elements ==> (AND comon_eles not_comon_eles)   !!!!!!!!!!!!!!
    # sinon             ===> not_comon_eles
    retour = ""
    if len(common_elements) > 0:
        retour += "(AND "
        retour += common_elements_str
        retour += not_common_elements_str
        retour += ")"
    else:
        retour += not_common_elements_str

    return retour

# listt = [['(z11)', '(z13)', '(z3)'], ['(not (z11))', '(not (z6))', '(z13)', '(z3)']]
# print(logic_factorization(listt))
# exit()

# list_of_clauses = [
#     ['del_0', 'del_2', 'del_3', 'del_4', 'add_5', 'del_8', 'del_9', 'del_10', 'del_11', 'del_12', 'add_13', 'add_15'],
#     ['add_0', 'del_2', 'del_3', 'del_4', 'del_5', 'add_6', 'add_7', 'del_8', 'del_9', 'add_10', 'add_11', 'del_12', 'del_13', 'add_14'],
#     ['add_0', 'add_1', 'del_2', 'add_3', 'del_4', 'add_5', 'del_6', 'del_7', 'add_8', 'add_9', 'add_10', 'del_11', 'del_12', 'del_13', 'add_15']
# ]

# list_of_clauses = [['(not (z13))', '(not (z5))', '(not (z7))', '(z6)', '(z4)', '(z2)'], ['(not (z0))', '(not (z13))', '(z6)', '(z7)', '(z4)', '(z2)'], ['(z11)', '(z13)', '(z3)', '(z4)', '(z2)'], ['(not (z11))', '(not (z5))', '(z13)', '(z3)', '(z4)', '(z6)', '(z4)', '(z2)'], ['(not (z11))', '(not (z6))', '(z13)', '(z3)', '(z4)', '(z2)']]


##### je veux que ça retourne, une liste 1D qui constitue une CONJUNCTION


# print(logic_factorization(list_of_clauses))
# exit()



def reformat_dict(D):
    uf = UnionFind()
    sublist_to_keys = defaultdict(set)


    # Build sublist_to_keys mapping
    for k in D:
        uf.find(k)  # Ensure parent is initialized
        for S in D[k]:
            T_S = tuple(S)
            sublist_to_keys[T_S].add(k)

    # Union keys that share sublists
    for keys_with_sublist in sublist_to_keys.values():
        keys_list = list(keys_with_sublist)
        for i in range(1, len(keys_list)):
            uf.union(keys_list[0], keys_list[i])

    # Collect groups
    groups = defaultdict(set)
    for k in D:
        leader = uf.find(k)
        groups[leader].add(k)

    # Reform the dictionary based on groups
    reformatted_dict = {}
    for group_keys in groups.values():
        group_keys = tuple(sorted(group_keys))  # Sort keys for consistent tuples
        # Find common sublists for the group
        common_sublists = set.intersection(*(set(map(tuple, D[k])) for k in group_keys))
        # Update each key's sublists by removing the common sublists
        for k in group_keys:
            D[k] = [sublist for sublist in D[k] if tuple(sublist) not in common_sublists]
        # Add the group and common sublists to the reformatted dictionary
        reformatted_dict[group_keys] = list(map(list, common_sublists))

    return reformatted_dict, D


def merge_and_cleanup(reformatted_dict, updated_D):
    # Merge the two dictionaries
    merged_dict = {}
    merged_dict.update(reformatted_dict)
    for k, v in updated_D.items():
        if v:  # Add only non-empty lists
            merged_dict[(k,)] = v

    # Remove entries with empty lists
    merged_dict = {k: v for k, v in merged_dict.items() if v}

    return merged_dict


def transfo_precond_effect_name_to_pddl(eff_name):

    if eff_name.split('_')[0] == 'del':
        eff_name = "(not (z"+eff_name.split('_')[1]+"))"
    else:
        eff_name = "(z"+eff_name.split('_')[1]+")"
    return eff_name


def sort_by_samples(lst):
    # Extract the number from the last element of each sublist and sort
    sorted_lst = sorted(lst, key=lambda x: int(x[-1].split(': ')[1]), reverse=True)
    return sorted_lst


def compute_total_and_save_histogram(data, filename="histogram.png"):
    # Extract the number of samples (last element of each sublist)
    sample_counts = [sublist[-1] for sublist in data]

    # Compute the total number of samples
    total_samples = sum(sample_counts)
    
    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(sample_counts) + 1), sample_counts, tick_label=[f"Sublist {i+1}" for i in range(len(sample_counts))])
    plt.xlabel("Sublists")
    plt.ylabel("Number of Samples")
    plt.title(f"Total Samples: {total_samples}")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Save the figure
    plt.savefig("actions/action_"+str(num_action)+"/"+filename, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

    return total_samples


def printsBranchesHistogramsAndFiltersOut(data_dict, num_action, output_dir="output", make_fig=False):
    """
    Generates and saves multiple histograms in a square grid layout in one big PNG file.
    Also filters and returns only sublists where the number of samples is >= the mean per effect.

    Parametersdata_dict:
    - : Dictionary where keys are effect names and values are lists of lists.
    - num_action: Action identifier to include in the output filename.
    - output_dir: Directory to save the output image.

    Returns:
    - filtered_data: Dictionary containing only sublists with samples >= mean per effect.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it doesn't exist


    # EX DATA is {'add_0': [['(not (z5))', '(z6)', '(not (z7))', '(not (z13))', 1], ['(z6)', '(z7)', '(not (z13))' .... ETC

    # EX DATA for DT2s is [['(not (z3))', '(not (z2))', '(z9)', '(not (z15))', '(not (z14))', '(not (z13))', '(not (z7))', '#samples: 3'], ['(not (z3))', '(z6)', .. ETC

    num_effects_or_preconds = 0 # eithe the num of cond effects (if we deal with dt1s) or the num of OR disjunctions (dt2s)
    effect_names_or_just_list_with_one_zero = None
    if type(data_dict) is dict: # we are dealing with DT1s
        effect_names_or_just_list_with_one_zero = list(data_dict.keys())
        num_effects_or_preconds = len(effect_names_or_just_list_with_one_zero)

    else: #then we are dealing with DT2s 
        num_effects_or_preconds = len(data_dict)
        effect_names_or_just_list_with_one_zero = [0]
    if num_effects_or_preconds == 0:
        print("No data to plot.")
        return {}

    # Dictionary to store filtered data (for output only, not affecting plots)
    filtered_data = {}

    if make_fig:
        # Compute grid size (square-like layout)
        grid_size = math.ceil(math.sqrt(num_effects_or_preconds))  # Closest square root
        rows, cols = grid_size, grid_size
        if (rows - 1) * cols >= num_effects_or_preconds:
            rows -= 1  # Reduce rows if unnecessary

        # Create figure with subplots
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flatten() if num_effects_or_preconds > 1 else [axes]

    total_samples_all = 0  # Total samples across all effects

    for i, effect_name_or_zero in enumerate(effect_names_or_just_list_with_one_zero):

        if make_fig:
            ax = axes[i]
        
        if type(data_dict) is dict:
            data = data_dict[effect_name_or_zero]
        else:
            data = data_dict
        # print("data_dictdata_dictdata_dictdata_dict")
        # print(data_dict)
        sample_counts = [sublist[-1] for sublist in data]  # Extract last element as samples

        if not sample_counts:
            continue  # Skip empty sublists

        # Compute mean
        mean_samples = np.mean(sample_counts)

        # Filter sublists where sample count >= mean (for output dictionary only)
        filtered_sublists = [sublist for sublist in data if sublist[-1] >= mean_samples]
        filtered_data[effect_name_or_zero] = filtered_sublists  # Store filtered sublists

        if make_fig:
            # Plot histogram with all data (not filtered)
            ax.bar(range(1, len(sample_counts) + 1), sample_counts,
                tick_label=[f"Sublist {j+1}" for j in range(len(sample_counts))])
            ax.set_xlabel("Sublists")
            ax.set_ylabel("Samples")
            ax.set_title(f"{effect_name_or_zero}: {sum(sample_counts)} (Mean: {mean_samples:.2f})")
            ax.grid(axis="y", linestyle="--", alpha=0.7)

        total_samples_all += sum(sample_counts)

    if make_fig:
        # Hide empty subplots (if any)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Set main title
        fig.suptitle(f"Total Samples for Action {num_action}: {total_samples_all}", fontsize=14, fontweight="bold")

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        # Save the figure
        output_filename = os.path.join(output_dir, f"num_action_{num_action}.png")
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved combined histogram to {output_filename}")

    if type(data_dict) is not dict:
        
        filtered_data = filtered_data[0]


    return filtered_data  # Return filtered sublists (not affecting plots)


def list_difference(list1, list2):
    """
    Returns the difference between two lists as a list of elements
    that are in one list but not in the other.
    """
    return list(set(list1) - set(list2)) + list(set(list2) - set(list1))


def remove_outer_and(expression):
    # Match the outer AND and remove it
    pattern = r'\(AND \((AND .*?)\)\)'
    simplified_expr = re.sub(pattern, r'(\1)', expression)
    return simplified_expr


def have_same_elements(list1, list2):
    """
    Check if two lists contain exactly the same elements, regardless of order.
    
    Args:
        list1 (list): First list.
        list2 (list): Second list.
    
    Returns:
        bool: True if both lists have the same elements, False otherwise.
    """
    return sorted(list1) == sorted(list2)



# def coverage_ids_precond(precond, low_lvl_actions_preconds):

#     #  precond like (z0)

#     # low_lvl_actions_preconds : all the preconds sets (ie 48) of the high lvl action we are interested of


#     # 


# returns the IDs of the trans the set of preconds covers
# def coverage_on_preconds(preconds_set):

#     # 

# # returns the IDs of the trans the set of effects covers
# def coverage_on_effects(effects_set):
#     # 

###############################################################
####################### END UTILS FUNCTIONS ###################
###############################################################



###############################################################
####################### BUILD THE X AND Y ####################
###############################################################


path_to_dataset = "/workspace/R-latplan/r_latplan_datasets/hanoi/hanoi_complete_clean_faultless_withoutTI/data.p"

loaded_data = load_dataset(path_to_dataset) # load dataset for the specific experiment
train_set_no_dupp = loaded_data["train_set_no_dupp_processed"]

dico_transitions_per_high_lvl_actions = {} # GROUP THE TRANSITIONS BY THEIR HIGH LVL ACTION

for ii, ele in enumerate(train_set_no_dupp): # loop over the train set (without dupplicate) # AND group the transitions into their respective High Level Actions
    if np.argmax(ele[2]) not in dico_transitions_per_high_lvl_actions:
        dico_transitions_per_high_lvl_actions[np.argmax(ele[2])] = {}
    if np.argmax(ele[1]) not in dico_transitions_per_high_lvl_actions[np.argmax(ele[2])]:
        dico_transitions_per_high_lvl_actions[np.argmax(ele[2])][np.argmax(ele[1])] = {
            "preprocessed" : ele[0],
        }

# print(len(dico_transitions_per_high_lvl_actions[21]))
# exit()


pos_preconds = np.loadtxt(base_dir+"/pos_preconds_aligned.csv", delimiter=' ', dtype=int)
neg_preconds = np.loadtxt(base_dir+"/neg_preconds_aligned.csv", delimiter=' ', dtype=int)

add_effs = np.loadtxt(base_dir+"/add_effs_aligned.csv", delimiter=' ', dtype=int)
del_effs = np.loadtxt(base_dir+"/del_effs_aligned.csv", delimiter=' ', dtype=int)

effects_set = []
for i in range(16):
    effects_set.append(f"add_{i}")
for i in range(16):
    effects_set.append(f"del_{i}")
preconds_perEff_perAction = {}

feature_names = []
for i in range(16):
    feature_names.append(f"(z_{i}_1)")
    feature_names.append(f"(z_{i}_0)")

preconds_names = []
for f in feature_names:
    preconds_names.append(format_precond(f.replace("(", "").replace(")", "")))



###############################################################
##############      X and Y FOR THE SECOND DT           #######
###############################################################


all_transitions_preconds = np.zeros((len(pos_preconds), len(pos_preconds[0])*2)) # (1469, 32)
all_Ys = np.zeros((len(pos_preconds), 22)) # (1469, 22)


for num_action in range(0, 22):
    for trans_id in dico_transitions_per_high_lvl_actions[num_action].keys():
        all_Ys[trans_id][num_action] = 1
        preconds_union = []
        preconds_union.extend(pos_preconds[trans_id])
        preconds_union.extend(neg_preconds[trans_id])
        all_transitions_preconds[trans_id] += preconds_union

#print(np.sum(np.sum(all_Ys, axis=1)))



all_transitions_preconds = pd.DataFrame(all_transitions_preconds)

new_columns = []
for i in range(16):
    new_columns.append(f"z_{i}")
for i in range(16):
    new_columns.append(f"not(z_{i})")

all_transitions_preconds.columns = new_columns[:len(all_transitions_preconds.columns)] # Use slicing to handle cases where X might have fewer columns


# print(all_transitions_preconds.head())
# exit()
new_entire_X = pd.DataFrame()
for i in range(16):
    # Create a new column in new_X
    new_column = []
    for index, row in all_transitions_preconds.iterrows():
        if row[f'z_{i}'] == 1:
            new_column.append(1)
        elif row[f'not(z_{i})'] == 1:
            new_column.append(0)
        else:
            new_column.append('?')  # Or any other representation you prefer for "otherwise"
    new_entire_X[f'z_{i}'] = new_column  # Assign the new column to the new DataFrame



new_entire_X.replace(1, "1", inplace=True)
new_entire_X.replace(0, "0", inplace=True)

onehot_encoder = OneHotEncoder(sparse=False, categories = [["1", "0", "?"] for i in range(16)])  # 'first' to drop the first category to avoid multicollinearity
onehot_encoded = onehot_encoder.fit_transform(new_entire_X) # Fit and transform the data
new_entire_X = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(['z_'+str(i) for i in range(16)]))

# removing the "?" columns
new_entire_X = new_entire_X.loc[:, ~new_entire_X.columns.str.contains(r'\?')]


###############################################################
################     end X and Y FOR THE NEW DT         #######
###############################################################



def extract_integers(s):
    return list(map(int, re.findall(r'\d+', s)))

##############################################################################
################  LOOP OVER EACH ACTION and learn the First tree #############
##############################################################################

for num_action in range(0, 22):



    if not os.path.exists("actions/action_"+str(num_action)):
        os.makedirs("actions/action_"+str(num_action))

    

    preconds_perEff_perAction[num_action] = {}

    ##################################################################################
    # Construct the X
    action_transitions_preconds = []
    for trans_id in dico_transitions_per_high_lvl_actions[num_action].keys():
        preconds_union = []
        preconds_union.extend(pos_preconds[trans_id])
        preconds_union.extend(neg_preconds[trans_id])
        action_transitions_preconds.append(preconds_union)
    X = np.array(action_transitions_preconds)



    X = pd.DataFrame(X)
    new_columns = []
    for i in range(16):
        new_columns.append(f"z_{i}")
    for i in range(16):
        new_columns.append(f"not(z_{i})")
    X.columns = new_columns[:len(X.columns)] # Use slicing to handle cases where X might have fewer columns
    X.index = list(dico_transitions_per_high_lvl_actions[num_action].keys())
    indices_with_nine = X[X.isin([9]).any(axis=1)].index
    X = X.drop(index=indices_with_nine)


    


    # Create a new DataFrame for the modified X matrix
    new_X = pd.DataFrame()
    for i in range(16):
        # Create a new column in new_X
        new_column = []
        for index, row in X.iterrows():
            if row[f'z_{i}'] == 1:
                new_column.append(1)
            elif row[f'not(z_{i})'] == 1:
                new_column.append(0)
            else:
                new_column.append('?')  # Or any other representation you prefer for "otherwise"
        new_X[f'z_{i}'] = new_column  # Assign the new column to the new DataFrame
    new_X.replace(1, "1", inplace=True)
    new_X.replace(0, "0", inplace=True)


    onehot_encoder = OneHotEncoder(sparse=False, categories = [["1", "0", "?"] for i in range(16)])  # 'first' to drop the first category to avoid multicollinearity
    onehot_encoded = onehot_encoder.fit_transform(new_X) # Fit and transform the data
    new_X = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(['z_'+str(i) for i in range(16)]))
    

    # removing the "?" columns
    new_X = new_X.loc[:, ~new_X.columns.str.contains(r'\?')]


    # END Construct the X
    ##################################################################################



    ##################################################################################
    # Construct the Y
    action_transitions_effects = []
    for trans_id in dico_transitions_per_high_lvl_actions[num_action].keys():
        effects_union = []
        effects_union.extend(add_effs[trans_id])
        effects_union.extend(del_effs[trans_id])
        action_transitions_effects.append(effects_union)
    Y = np.array(action_transitions_effects)
    Y = pd.DataFrame(Y)
    new_columns = []
    for i in range(16):
        new_columns.append(f"add_{i}")
    for i in range(16):
        new_columns.append(f"del_{i}")
    Y.columns = new_columns[:len(Y.columns)] # Use slicing to handle cases where X might have fewer columns
    Y.index = list(dico_transitions_per_high_lvl_actions[num_action].keys())
    Y = Y.drop(index=indices_with_nine)


    # END Construct the Y
    ##################################################################################



    ##################################################################################
    # Train the first classifier
    X_train, X_test, Y_train, Y_test = train_test_split(new_X, Y, test_size=0.2, random_state=42)
    clf = MultiOutputClassifier(DecisionTreeClassifier(criterion='entropy', random_state=42))
    clf.fit(X_train, Y_train)
    # Tester le modèle
    Y_pred = clf.predict(X_test)
    # Évaluer la performance du modèle (par exemple, par une accuracy moyenne)
    accuracy = np.mean(Y_pred == Y_test)
    #print(f"Accuracy: {accuracy}")
    # END Train the first classifier
    ##################################################################################



    ##################################################################################
    # Print or/and return the DT branches (FIRST DT)
    for idx, estimator in enumerate(clf.estimators_):
        effect_name = effects_set[idx]

        # #### PRINTING
        # plt.figure(figsize=(12, 12))
        # plot_tree(estimator, 
        #         feature_names=feature_names, 
        #         label= 'all',
        #         class_names=[f'Class {k}' for k in range(2)], 
        #         filled=True, 
        #         rounded=True,
        #         impurity = True
        #         )
        # say_one_positive_node = ""
        # clf__ = estimator
        # # Ugly paths are ALL the path of the current tree (one tree per effect)
        # ugly_paths = get_rules(clf__, feature_names, None)
        # if len(ugly_paths) == 1 and len(ugly_paths[0]) == 1:
        #     one_node = True
        #     summ = int(Y_train[effect_name].sum())
        #     if summ > 0:
        #         say_one_positive_node = "CLASS 1"
        #         print("num_action is {}, effect is {}, CLASS 1:".format(str(num_action), str(effect_name)))
        #     else:
        #         say_one_positive_node = "CLASS 0"
        # plt.title(f"Decision Tree for Effect {effect_name}, {say_one_positive_node}", fontsize = 40)
        # file_name = f"DTs_action_{str(num_action)}/decision_tree_effect_{effect_name}.png"
        # plt.savefig(file_name, format="png", dpi=300)  

        clf__ = estimator
        ugly_paths = get_rules(clf__, feature_names, None)



        one_node = False
        the_one_node_is_true = False
        add_stuff = False
    
        if len(ugly_paths) == 1 and len(ugly_paths[0]) == 1:
            one_node = True
            summ = int(Y_train[effect_name].sum())
            if summ > 0:
                ugly_paths[0][0].replace('0', '1')

                the_one_node_is_true = True
        beauty_paths = []

        if not one_node:
            add_stuff = True    
            for pathh in ugly_paths:
                if 'class is: 0' in pathh[-1]:
                    continue
                elif 'class is: 1' in  pathh[-1]: # and len(pathh[:-1]) > 2:
                    tmp = []
                    integer = int(''.join(x for x in str(pathh[-1].split(", ")[1]) if x.isdigit()))
                    for cond in pathh[:-1]:
                        if ">" in cond:
                            tmp.append(format_precond(cond.split(" > ")[0].replace("(", "").replace(")", "")))
                        else: # "<=" in cond
                            tmp.append(format_precond(cond.split(" <= ")[0].replace("(", "").replace(")", ""), reverse=True))
                            #tmp.append(format_precond(cond.split(" <= ")[0].replace("(", "").replace(")", "").replace("not", "")))
                            #break
                    tmp.append("#samples: "+str(integer))
                    if len(tmp) > 0:
                        
                        beauty_paths.append(tmp)


            preconds_perEff_perAction[num_action][effect_name] = beauty_paths
            
        elif the_one_node_is_true:
            tmp = []

            integer = int(''.join(x for x in str(ugly_paths[0][0].split(", ")[1]) if x.isdigit()))
            tmp.append("#samples: "+str(integer))
            beauty_paths.append(tmp)
            add_stuff = True
            preconds_perEff_perAction[num_action][effect_name] = beauty_paths
        #print("=" * 50)  # Separator between trees




    # remove dupplicates from preconds groups
    for eff_n, list_lists in preconds_perEff_perAction[num_action].items():

        for ijijijij, l in enumerate(list_lists):
            nb_samples = extract_integers(l[-1])[0]

            l = list(set(l[:-1])) # remove dupplicates
            #preconds_perEff_perAction[num_action][eff_n][ijijijij] = l
            # remove contradictions from preconds groups
            l_ = l.copy()
            dones = []
            for ele in l:
                if opposite(ele) in dones:
                    l_.remove(ele)
                    l_.remove(opposite(ele))
                dones.append(ele)
            preconds_perEff_perAction[num_action][eff_n][ijijijij] = l_
            preconds_perEff_perAction[num_action][eff_n][ijijijij].append(nb_samples)


 
    # END Print or/and return the DT branches (FIRST DT)
    ##################################################################################

    # print(" LES KEYYYYSSSSSSSSS ")
    # print(preconds_perEff_perAction[num_action].keys())

    # print(preconds_perEff_perAction[num_action]["add_0"])

    

    ##################################################################################
    # Print or/and return the DT branches (FIRST DT)
    # for idx, estimator in enumerate(clf.estimators_):

    # prints histograms of class 1 leaves AND filters out less important leaves
    if filter_out_dt1:
        preconds_perEff_perAction[num_action] = printsBranchesHistogramsAndFiltersOut(preconds_perEff_perAction[num_action], num_action, output_dir = "output_dt1s_histos", make_fig = False)

    # for effect_name in preconds_perEff_perAction[num_action].keys():
    #     compute_total_and_save_histogram(preconds_perEff_perAction[num_action][effect_name], effect_name+".png")

    # if num_action == 2:
    #     print("preconds_perEff_perAction[2]['del_5']")
    #     print(preconds_perEff_perAction[2]["del_5"])
    #     print()
    #     print()
    #     print()
    #     print(preconds_perEff_perAction[2])
    #     exit()
##############################################################################
##############  END LOOP OVER EACH ACTION and learn the First tree   #########
##############################################################################


def add_parenthesis(liste):
    liste_re = []
    for ele in liste:
        liste_re.append("("+ele+")")
    return liste_re


##############################################################################
###########################  WRITE THE PDDL   ################################
#######################  and LEARN THE SECOND TREE ###########################
##############################################################################
with open(base_dir+"/confs/"+dir_name+"/"+"domainCondBIS.pddl", "w") as f:


    f.write("(define (domain latent)\n")
    f.write("(:requirements :negative-preconditions :strips :conditional-effects)\n")
    f.write("(:types\n")
    f.write(")\n")
    f.write("(:predicates\n")

    for i in range(X.shape[1]//2):
        f.write("(z"+str(i)+" )\n")
    #feature_names = [f"precondition_{i}" for i in range(X.shape[1])]
    f.write(")\n")


    #dico_disjunctions_per_high_lvl_action_ = dico_disjunctions_per_high_lvl_action()
    #dico_disjunctions_simple_per_high_lvl_action_ = dico_disjunctions_simple_per_high_lvl_action()

    dico_disjunctions_preconds_dt = {}

    for num_action in range(0, 22):

        # if num_action != 1:
        #     continue

        dico_disjunctions_preconds_dt[num_action] = []
        # thedico = {
        #     'add_0': 
        #         [
        #             ['(not (z13))', '(z6)', '(not (z7))', '(not (z5))'], 
        #             ['(not (z13))', '(z6)', '(z7)'], 
        #         ], 
        #     'add_1': [], 
        #     'add_2': [
        #         ['(not (z2))', '(z9)', '(not (z14))', '(not (z6))', '(not (z15))', '(not (z7))', '(z4)'], 
        #         ['(not (z2))', '(z9)', '(not (z14))', '(z6)', '(z5)'], 
        #         ['(not (z2))', '(not (z9))']
        #     ], 
        #     'add_3': [], 
        #     'add_4': [
        #         ['(not (z2))', '(not (z9))']
        #     ], 
        #     'add_5': [
        #         ['(not (z2))', '(not (z9))'],
        #         ['(not (z13))', '(z6)', '(not (z7))', '(not (z5))']
        #     ]
        # }

        merged_dict = factorize_dict(preconds_perEff_perAction[num_action], num_action)
        f.write("(:action a"+str(num_action)+"\n")
        f.write("   :parameters ()\n")

        #f.write("   :precondition ()\n") # IF NOT PRECONDITIONS


        new_entire_X_ = new_entire_X.copy()
        liste_preconds_to_remove = []
        liste_preconds_to_remove_bis = []


        #### store of preconds that are already in the cond effects of the high lvl action
        all_preconds_of_high_lvl = []
        for eff_group, preconds_sets in merged_dict.items():
            for prec_set in preconds_sets:
                for precc in prec_set:
                    all_preconds_of_high_lvl.append(precc)


        all_preconds_of_high_lvl = list(set(all_preconds_of_high_lvl))

        # for prec in preconds_names:
        #     if prec not in all_preconds_of_high_lvl:
        #         liste_preconds_to_remove_bis.append(prec)
        #         #[['(not (z13))', '(not (z5))', '(not (z7))', '(z6)'], ['(not (z0))', ......


        all_preconds_NOT_in_cond_effects = list_difference(preconds_names, all_preconds_of_high_lvl)

        if x_atoms_from_cond_effects_dt2:
            if len(all_preconds_of_high_lvl) > 0:
                new_entire_X_ = new_entire_X_.drop(columns=[reverse_format(prec_) for prec_ in all_preconds_NOT_in_cond_effects])
                

        elif x_atoms_outisde_of_cond_effects_dt2:
            if len(all_preconds_of_high_lvl) > 0:
                new_entire_X_ = new_entire_X_.drop(columns=[reverse_format(prec_) for prec_ in all_preconds_of_high_lvl])

        elif x_all_atoms_dt2:
            pass

        # print("LAAAA")
        # exit()

        beauty_paths = []

        if not new_entire_X_.empty:

            #print("action {} entire NO empty".format(num_action))



            # For each action, do the DT of preconds for knowning if apply the action or not

            clf2 = DecisionTreeClassifier(criterion='entropy', random_state=42)
            clf2.fit(new_entire_X_, all_Ys[:, num_action])


            ### Routine to count the # blue leaves
            # tree = clf2.tree_
            # # Dictionary to store count of leaves by number of samples
            # leaf_samples_dict = defaultdict(int)
            # for node in range(tree.node_count):
            #     if tree.children_left[node] == -1 and tree.children_right[node] == -1:  # Check if it's a leaf
            #         predicted_class = np.argmax(tree.value[node])  # Get the predicted class
            #         if predicted_class == 1:  # Only count leaves with class 1
            #             num_samples = int(tree.n_node_samples[node])  # Get the number of samples
            #             leaf_samples_dict[num_samples] += 1  # Increment count for that sample size
            ## Convert defaultdict to a regular dict
            # leaf_samples_dict = dict(leaf_samples_dict)
            # print("Leaves with class 1, grouped by number of samples:", leaf_samples_dict)


            # if num_action == 5:
            #     print("new_entire_X_")
            #     print(list(new_entire_X_.columns))
            #     print("feature_names")
            #     print(feature_names)
            #     print("add_parenthesis(list(new_entire_X_.columns))")
            #     print(add_parenthesis(list(new_entire_X_.columns)))
            #     exit()


    


            ugly_paths = get_rules(clf2, add_parenthesis(list(new_entire_X_.columns)), None)


            # print("UGLUY PATHSSSSSSSSSSSSSSSSS")
            # print(ugly_paths)
     
            # if num_action == 5:
            #     print("ugly_pathsugly_pathsugly_paths 5")
            #     print(ugly_paths)
            #     exit()

            # if num_action == 21:

            #     # print("list(new_entire_X_.columns)")
            #     # print(list(new_entire_X_.columns))
            #     # print("feature_names")


            #     plt.figure(figsize=(72, 72))
            #     plot_tree(clf2, 
            #             feature_names=add_parenthesis(list(new_entire_X_.columns)), 
            #             label= 'all',
            #             class_names=[f'Class {k}' for k in range(2)], 
            #             filled=True, 
            #             rounded=True,
            #             impurity = True
            #             )
            #     say_one_positive_node = ""
            #     plt.title(f"Decision Tree for Action {num_action}, {say_one_positive_node}", fontsize = 40)
            #     file_name = f"decision_tree_HIGH_LEVEL_{num_action}.png"
            #     plt.savefig(file_name, format="png", dpi=300)  

            # # exit()


            one_node = False
            the_one_node_is_true = False
            add_stuff = False
            if len(ugly_paths) == 1 and len(ugly_paths[0]) == 1:
                one_node = True
                summ = int(all_Ys[:, num_action].sum()) # check if class is 1, ie is the sum is > 0
                # means that the effect was "1" (ie applied) at least once
                if summ > 0:
                    ugly_paths[0][0].replace('0', '1')
                    the_one_node_is_true = True
    
            #beauty_paths = []
            if not one_node:
                add_stuff = True
                for pathh in ugly_paths:
                    if 'class is: 0' in pathh[-1]:
                        continue
                    elif 'class is: 1' in pathh[-1]: # and len(pathh[:-1]) > 2:
                        tmp = [] #{}
                        integer = int(''.join(x for x in str(pathh[-1].split(", ")[1]) if x.isdigit()))
                        #tmp[integer] = []
                        for cond in pathh[:-1]:
                            if ">" in cond:
                                #print(">>>>>>>>>")
                                #print("cond.split(" > ")[0].replace("(", "").replace(")", "")")
                                # print(cond.split(" > ")[0].replace("(", "").replace(")", ""))
                                # print(format_precond(cond.split(" > ")[0].replace("(", "").replace(")", "")))
                                # exit()
                                tmp.append(format_precond(cond.split(" > ")[0].replace("(", "").replace(")", "")))
                            else: # "<=" in cond
                                tmp.append(format_precond(cond.split(" <= ")[0].replace("(", "").replace(")", ""), reverse=True))
                                #tmp.append(format_precond(cond.split(" <= ")[0].replace("(", "").replace(")", "").replace("not", "")))
                                #break
                        tmp.append("#samples: "+str(integer))
                        if len(tmp) > 0:

                            beauty_paths.append(tmp)

                #preconds_perEff_perAction[num_action][effect_name] = beauty_paths

            elif the_one_node_is_true:
                tmp = []
                integer = int(''.join(x for x in str(ugly_paths[0].split(", ")[1]) if x.isdigit()))
                tmp.append("#samples: "+str(integer))
                beauty_paths.append(tmp)
                add_stuff = True
                #preconds_perEff_perAction[num_action][effect_name] = []
        else:
            pass
            #print("action {} entire IS  empty".format(num_action))
   

            
        if len(beauty_paths) > 0:


            # liste_preconds_to_remove_bis_formatted = []

            # for ell in list(new_entire_X_.columns):
            #     # print("action is {}".format(str(num_action)))
            #     # print("ell is {}".format(str(ell)))
            #     liste_preconds_to_remove_bis_formatted.append(format_precond(ell))

            beauty_paths_sorted = sort_by_samples(beauty_paths)

            # for bb in beauty_paths_sorted:
            #     if "#samples: 1" in bb[-1] or "#samples: 2" in bb[-1]:
            #         beauty_paths_sorted.remove(bb)

            for ijijijij, l in enumerate(beauty_paths_sorted):
                nb_samples = extract_integers(l[-1])[0]


                l = list(set(l[:-1]))
                l_ = l.copy()
                dones = []
                for ele in l[:-1]:
        
                    if opposite(ele) in dones:
                        l_.remove(ele)
                        l_.remove(opposite(ele))
                    dones.append(ele)

                beauty_paths_sorted[ijijijij] = l_
                beauty_paths_sorted[ijijijij].append(nb_samples)


            beauty_paths_sorted_pruned = []

            for path in beauty_paths_sorted:

                add_it = True

                for to_rem in liste_preconds_to_remove_bis:

                    if to_rem in path:

                        add_it = False
                        

                if add_it:
                    beauty_paths_sorted_pruned.append(path) 

            # if num_action == 5:
            #     print("beauty_paths_sorted_prunedbeauty_paths_sorted_pruned")
            #     print(beauty_paths_sorted_pruned)
            #     exit()

            beauty_paths_sorted = beauty_paths_sorted_pruned

            # print("SIZE beauty_paths_sorted {}".format(str(len(beauty_paths_sorted))))
            # continue

            if filter_out_dt2:
                #print(beauty_paths_sorted)
                beauty_paths_sorted = printsBranchesHistogramsAndFiltersOut(beauty_paths_sorted, num_action, output_dir = "outputs_dt2s_histos", make_fig = False)
           

            #beauty_paths_sorted = [1,2,3,4,5,6,7]
            for c1, l1 in enumerate(beauty_paths_sorted):
                l1 = l1[:-1]
                for c2 in range(c1+1, len(beauty_paths_sorted)):
                    l2 = beauty_paths_sorted[c2][:-1]
                    #print("l1 {} l2 {}".format(l1, l2))

                    assert sorted(l1) != sorted(l2)


            total_samples_in_beauty_paths = 0
            if len(beauty_paths_sorted) > 0:

                # CASE : precond is an OR of the set of all high lvl preconds 
                f.write("   :precondition (OR ")

                for liste in beauty_paths_sorted:
                    integer = int(''.join(x for x in str(liste[-1]) if x.isdigit()))
                    total_samples_in_beauty_paths += integer
                    tmp_str = "(AND "
                    for ele in liste[:-1]:
                        #ele = transfo_precond_effect_name_to_pddl(ele)
                        tmp_str += " "+ele 
                    tmp_str += ")"
                    f.write(" "+tmp_str)
                f.write(")\n") # closing the (



                # print("ACTION a{}".format(str(num_action)))
                # print("number of disjunctions in the OR is {}".format(str(len(beauty_paths_sorted))))
                # print("#samples covered by the disjunctions is {}".format(str(total_samples_in_beauty_paths)))
                # print("nber of transitions is {}".format(str(len(dico_transitions_per_high_lvl_actions[num_action].keys()))))



            else:
                f.write("   :precondition ()\n") # IF NOT PRECONDITIONS
        else:

            f.write("   :precondition ()\n") # IF NOT PRECONDITIONS



        f.write("   :effect (and\n")

        # ##### CODE FOR HAVING THE PRECONDS FOR EACH EFFECT
        # for eff_name, preconds_sets in preconds_perEff_perAction[num_action].items():
        #     two_tabs_space  = "         "
        #     str_preconds = ""
        #     if len(preconds_sets) > 0:
        #         for precond_set in preconds_sets:
        #             # if len(precond_set) > 1:
        #             #     print("KKKKKKK")
        #             #     print(precond_set)
        #             #     exit()
        #             tmp_str = two_tabs_space+"(when "
        #             if len(precond_set) == 1:
        #                 tmp_str += precond_set[0]+"\n"
        #             else:
        #                 sub_str = "(and "+" "
        #                 close_preconds = ""
        #                 for precond in precond_set:
        #                     close_preconds += precond + " "
        #                 sub_str += close_preconds
        #                 sub_str += ")\n"

        #                 tmp_str += sub_str

        #             if eff_name.split('_')[0] == 'del':
        #                 eff_name = "(not (z"+eff_name.split('_')[1]+"))"
        #             else:
        #                 eff_name = "(z"+eff_name.split('_')[1]+")"
        #             tmp_str += two_tabs_space+eff_name + "\n"
        #             tmp_str += two_tabs_space+")\n"
        #             str_preconds += tmp_str
        #     f.write(str_preconds)
        # if there is a contradiction... 
        # 


        ##### CODE FOR HAVING THE PRECONDS FOR GROUP OF EFFECTS

        for eff_group, preconds_sets in merged_dict.items():

            # factorize_dt1
            if factorize_dt1:
                str_preconds = ""
                if len(preconds_sets) > 0:



                    two_tabs_space  = "         "
                    tmp_str = two_tabs_space+"(when "
                    


                    if len(preconds_sets) == 1:
                        
                        precond_set = preconds_sets[0]

                        sub_str = "(and "+" "
                        close_preconds = ""
                        for precond in precond_set:
                            close_preconds += precond + " "
                        sub_str += close_preconds
                        sub_str += ")\n"

                        tmp_str += sub_str

                    else:

                        factorized_ = logic_factorization(preconds_sets)
                        tmp_str += factorized_ + "\n"

                    str_before = ''
                    str_after = ''
                    if len(list(eff_group)) > 1:
                        str_before = two_tabs_space + two_tabs_space + "(and \n"
                        str_after = two_tabs_space + two_tabs_space + ")\n"

                    tmp_group_of_effects = ''
                    for eff_name in  list(eff_group):

                        eff_name = transfo_precond_effect_name_to_pddl(eff_name)
                        
                        tmp_group_of_effects += two_tabs_space+ two_tabs_space + eff_name + "\n"

                    tmp_str += str_before + tmp_group_of_effects + str_after
                    tmp_str += two_tabs_space+")\n"
                
                    
                    str_preconds += tmp_str #remove_outer_and(tmp_str)


                else: # precond group is an empty list, therefore the effect must always be applied

                    eff_name = list(eff_group)[0]
                        
                    eff_name = transfo_precond_effect_name_to_pddl(eff_name)
                    #print("eff_name {}".format(str(eff_name)))
                    #tmp_group_of_effects += two_tabs_space+eff_name + "\n"
                    str_preconds += two_tabs_space + eff_name + "\n"

                f.write(str_preconds)

            else:
                    
                two_tabs_space  = "         "
                str_preconds = ""
                if len(preconds_sets) > 0:
                    for precond_set in preconds_sets:

                        tmp_str = two_tabs_space+"(when "
                        if len(precond_set) == 1:
                            tmp_str += precond_set[0]+"\n"
                        else:
                            sub_str = "(and "+" "
                            close_preconds = ""
                            for precond in precond_set:
                                close_preconds += precond + " "
                            sub_str += close_preconds
                            sub_str += ")\n"

                            tmp_str += sub_str


                        # eff_group
                        str_before = ''
                        str_after = ''
                        if len(list(eff_group)) > 1:
                            str_before = two_tabs_space + two_tabs_space + "(and \n"
                            str_after = two_tabs_space + two_tabs_space + ")\n"
                        #tmp_str += str_before
                        tmp_group_of_effects = ''
                        for eff_name in  list(eff_group):

                            eff_name = transfo_precond_effect_name_to_pddl(eff_name)
                            
                            tmp_group_of_effects += two_tabs_space+ two_tabs_space + eff_name + "\n"

                        tmp_str += str_before + tmp_group_of_effects + str_after
                        tmp_str += two_tabs_space+")\n"

                        str_preconds += tmp_str

                else: # precond group is an empty list, therefore the effect must always be applied

                    eff_name = list(eff_group)[0]
                        
                    eff_name = transfo_precond_effect_name_to_pddl(eff_name)
                    #print("eff_name {}".format(str(eff_name)))
                    #tmp_group_of_effects += two_tabs_space+eff_name + "\n"
                    str_preconds += two_tabs_space + eff_name + "\n"
                f.write(str_preconds)



        f.write("   )\n")
        f.write(")\n")

        # if num_action == 2:

        #     exit()
    
    f.write(")\n")

