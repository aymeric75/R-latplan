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
from collections import defaultdict
import os
import sys
from itertools import combinations, islice
from multiprocessing import Pool, cpu_count, Array
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
from sympy import symbols, And, Or, Not, simplify_logic
import itertools
from tqdm import tqdm
import multiprocessing
import pickle
import copy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
import math
from collections import Counter
from typing import List
from math import comb
from itertools import islice
import textwrap
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances
import argparse



parser = argparse.ArgumentParser(description="A script cluster low level actions of R-latplan")

parser.add_argument('--base_dir', default=None, type=str, help='Optional: Base path of the current experiment', required=False)

parser.add_argument('--data_folder', default=None, type=str, help='Optional: Base path of the current experiment data', required=False)


parser.add_argument('--clustering_base_data', default=None, type=str, choices=['only_preconds', 'only_effects', 'both'], help='Optional: indicates which type of data are used for the clustering', required=True)

parser.add_argument('--clustering_with_penalty', default=None, type=str, help='Optional: indicates if we use or not the penalty in Jaccard distance', required=True)

parser.add_argument('--specific_whens', default="False", type=str, help='Optional: indicates if we use or not the specific_whens ', required=False)



args = parser.parse_args()

base_dir = args.base_dir
data_folder = args.data_folder


two_tabs_space  = "         "



precondition_only_entailing = True #True # tells if we consider, in the precondition clause, only the sets that entail other sets
filter_out_condEffects = False #True # tells if we filter out from the preconditions of the conditional effects, the atoms that are already
# present in the precondition clause

def load_dataset(path_to_file):
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

            


def intersection_of_lists(lists: List[List[str]]) -> List[str]:
    if not lists:
        return []
    # Start with the first list as the base set
    common_elements = set(lists[0])
    # Intersect with each subsequent list
    for lst in lists[1:]:
        common_elements.intersection_update(lst)
    return list(common_elements)


def transform_atoms_list(atoms_list):
    transformed = []
    for sublist in atoms_list:
        parts = []
        for atom in sublist:
            if atom.startswith('(not (z') and atom.endswith('))'):
                # Example: (not (z1)) -> ~z1
                parts.append('~' + atom[6:-2])
            elif atom.startswith('(z') and atom.endswith(')'):
                # Example: (z1) -> z1
                parts.append(atom[1:-1])
            else:
                raise ValueError(f"Unexpected atom format: {atom}")
        combined = '{ ' + ' '.join(parts) + ' }'
        transformed.append(combined)
    return transformed



def fromBinaryMatrixToStrings(sorted_combinationsMatrix_int):

    combinations_list = []
    for row in sorted_combinationsMatrix_int:
        selected_atoms = [atoms[i] for i, val in enumerate(row) if val == 1]
        combinations_list.append(selected_atoms)
    #print(combinations_list)

    
    combinations_list_tranfo = transform_atoms_list(combinations_list)
    # for ele in combinations_list_tranfo:
    #     print(ele)
    
    return combinations_list_tranfo



def bron_kerbosch(R, P, X, is_compatible, results):
    if not P and not X:
        results.append(R)
        return

    for v in list(P):
        new_R = R + [v]
        new_P = [u for u in P if is_compatible(v, u)]
        new_X = [u for u in X if is_compatible(v, u)]
        bron_kerbosch(new_R, new_P, new_X, is_compatible, results)
        P.remove(v)
        X.append(v)


# def common_ones(a: np.ndarray, b: np.ndarray) -> int:
#     """
#     Returns the number of positions where both binary vectors have 1s.
    
#     Parameters:
#     - a (np.ndarray): First binary vector.
#     - b (np.ndarray): Second binary vector.

#     Returns:
#     - int: Count of positions with 1s in both vectors.
#     """
#     if a.shape != b.shape:
#         raise ValueError("Input vectors must have the same shape.")
#     return int(np.sum(np.bitwise_and(a, b)))





#  returns the "mask" (actually binary array) that tells where two arrays contradict (by comparing the two halves of each)
def compare_crossed_halves(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    n = arr1.shape[0]
    half = n // 2
    retour = np.zeros(n, dtype=bool)
    retour[:half] = arr1[:half] & arr2[half:]
    retour[half:] = arr2[:half] & arr1[half:]
    return retour



def matrix_cross_compare(matrix: np.ndarray) -> np.ndarray:
    # Validate input
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D binary matrix")

    n_cols = matrix.shape[1]
    theretour = np.zeros(n_cols, dtype=bool)

    # Iterate over all unique pairs of distinct rows
    for i, j in combinations(range(matrix.shape[0]), 2):
        arr1 = matrix[i]
        arr2 = matrix[j]
        comp = compare_crossed_halves(arr1, arr2)
        theretour |= comp  # OR with the accumulated result

    return theretour




def format_literals_1(lst):
    # Step 1: Parse and categorize literals
    positives = []
    negatives = []

    for item in lst:
        if item.startswith('(not'):
            num = int(item.split('z')[1].strip(')'))
            negatives.append((num, f'~z{num}'))
        else:
            num = int(item.split('z')[1].strip(')'))
            positives.append((num, f'z{num}'))

    # Step 2: Sort by numerical order
    positives.sort()
    negatives.sort()

    # Step 3–5: Combine and format the final string
    combined = [literal for _, literal in positives + negatives]
    result = '  '.join(combined)
    return result

def sort_clause_string(clause_str):
    # Remove curly braces and split
    literals = clause_str.strip('{} ').split()
    
    # Sort positive and negative separately
    positives = sorted([lit for lit in literals if not lit.startswith('~')], key=lambda x: int(x[1:]))
    negatives = sorted([lit for lit in literals if lit.startswith('~')], key=lambda x: int(x[2:]))
    
    # Join into a string
    return '  '.join(positives + negatives)


def format_literals_2(data):

    result = [sort_clause_string(clause) for clause in data]

    return result


def all_unique_combinations(lst):
    result = []
    for r in range(1, len(lst) + 1):
        result.extend(combinations(lst, r))
    return result










def chunked_iterable(iterable, size):
    """Split iterable into chunks of given size."""
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])

def process_comb_chunk_with_base_paths(args):
    chunk, base_paths = args
    return [set(base_paths + list(comb)) for comb in chunk]

def parallel_preconds_with_progress(all_combs, base_paths, chunk_size=1000):
    chunks = list(chunked_iterable(all_combs, chunk_size))  # Ensure materialization
    total_chunks = len(chunks)

    args_list = [(chunk, base_paths) for chunk in chunks]
    preconds_paths_for_this_when = []
    cpu_count = multiprocessing.cpu_count()  # Use all available CPUs
    with multiprocessing.Pool(processes=cpu_count) as pool:
        for result in tqdm(pool.imap(process_comb_chunk_with_base_paths, args_list), total=total_chunks, desc="Processing combinations"):
            preconds_paths_for_this_when.extend(result)

    return preconds_paths_for_this_when


###################
###################   I. CONSTRUCT THE DICO OF HIGH LVL VS LOW LVL ACTIONS  (use the domain.pddl)
###################

path_to_dataset = data_folder + "/data.p"
loaded_data = load_dataset(path_to_dataset)
train_set_no_dupp = loaded_data["train_set_no_dupp_processed"]

latent_size = 0
nber_llas = len(train_set_no_dupp[0][1])
nber_hlas = len(train_set_no_dupp[0][2])



dico_lowlvl_highlvl = {} 
for ii, ele in enumerate(train_set_no_dupp):
    if np.argmax(ele[1]) not in dico_lowlvl_highlvl:
        dico_lowlvl_highlvl[np.argmax(ele[1])] = np.argmax(ele[2])



dico_highlvlid_lowlvlactions = {}

#domainfile="domain_ORIGINAL_NO_DUPP.pddl"
domainfile=base_dir + "/domain.pddl"
problemfile=base_dir + "/problem.pddl"
translate_path = os.path.join("/workspace/R-latplan", "downward", "src", "translate")
sys.path.insert(0, translate_path)

try:

    import FDgrounder
    from FDgrounder import pddl_parser as pddl_pars

    task = pddl_pars.open(
    domain_filename=domainfile, task_filename=problemfile) 
    nb_total_actions = len(task.actions) + len(task.indices_actions_no_effects)
    cc_normal_actions = 0


    for pre in task.predicates:
        if "z" in pre.name:
            latent_size += 1

    for trans_id in range(nb_total_actions):


        if trans_id in task.indices_actions_no_effects:
            pass 
        else:
            act = task.actions[cc_normal_actions]

            low_lvl_name_clean = act.name.split("-")[0].split("+")[0]

            #highlvlid = dico_lowlvl_highlvl[trans_id]
            # print("low_lvl_name_clean")
            # print(low_lvl_name_clean)
            highlvlid = dico_lowlvl_highlvl[int(low_lvl_name_clean[1:])]

        

            if highlvlid not in dico_highlvlid_lowlvlactions:
                dico_highlvlid_lowlvlactions[highlvlid] = {}

            if trans_id not in dico_highlvlid_lowlvlactions[highlvlid]:
                dico_highlvlid_lowlvlactions[highlvlid][trans_id] = {"preconds": [], "effects": []}


            for precond in list(act.precondition.parts):
                f_name_precond = friendly_name(precond)
                dico_highlvlid_lowlvlactions[highlvlid][trans_id]["preconds"].append(f_name_precond)

            for eff in act.effects:
                f_name_eff = friendly_name(eff.literal)
                dico_highlvlid_lowlvlactions[highlvlid][trans_id]["effects"].append(f_name_eff)
            # friendly_name
            cc_normal_actions += 1

finally:
    # Restore sys.path to avoid conflicts
    sys.path.pop(0)

two_tabs_space  = "         "

# dico_highlvlid_lowlvlactions 



###################
###################   II. PREPARE THE STRING TO WRITE INTO THE PDDL FILE (whole_actions_str)
###################




atoms = [] # = U
for i in range(latent_size):
    atoms.append(f"(z{i})")
for i in range(latent_size):
    atoms.append(f"(not (z{i}))")


whole_actions_str = ""
paths_preconditions_all_high_lvl_actions_all_clusters = {}
paths_effects_all_high_lvl_actions_all_clusters = {}
all_clusters = {}
all_clusters_preconds = {}


high_lvl_action_str_gen = ""

for num_action in range(0, nber_hlas):

    print("action is {}".format(str(num_action)))

    paths_preconditions_all_high_lvl_actions_all_clusters[num_action] = {}
    paths_effects_all_high_lvl_actions_all_clusters[num_action] = {}

    precondition_part = "" 
    intersection_part = ""
    or_clause = ""
    and_str = ""

    last_current_time = time.time()


    finat_set = np.array([])
    S = []
    E = [] 
    U_pre = []


    for thecounter, (lowlvlkey, dico_vals) in enumerate(dico_highlvlid_lowlvlactions[num_action].items()):
        S.append(dico_vals["preconds"])
        E.append(dico_vals["effects"])
        for atom in dico_vals["preconds"]:
            U_pre.append(atom)


    U_pre = list(set(U_pre))

    # 1 -- taking the intersection of all the preconditions set of the high lvl action
    I = intersection_of_lists(S)

    if len(I) > 0:
        and_str += "(AND "
        if len(I) == 1:
            and_str += I[0]
        else:
            and_str += "( "
            for el in I:
                and_str += el + " "
            and_str += " )"





    ##### UNIT TEST 3: check the precondition_and part
    # first uncomment the fake S introduced in UNIT TEST 2 and the print(S) below
    # 3-4 --
    for s in S:
        for i in I:
            if i in s:
                s.remove(i)

    # l. 6
    for u in U_pre:
        for i in I:
            if i in U_pre:
                U_pre.remove(i)


    binary_S = np.zeros((len(S), latent_size*2), dtype=np.uint8)
    binary_S = pd.DataFrame(binary_S)
    binary_S.columns = atoms
    for idx, row in binary_S.iterrows():
        for col in S[idx]:
            if col in binary_S.columns:
                binary_S.at[idx, col] = 1
    binary_S = binary_S.to_numpy()

    print("binary_S.shape")
    print(binary_S.shape) # (1162, 50)

    binary_Effects = np.zeros((len(E), latent_size*2), dtype=np.uint8)
    binary_Effects = pd.DataFrame(binary_Effects)
    binary_Effects.columns = atoms
    for idx, row in binary_Effects.iterrows():
        for col in E[idx]:
            if col in binary_Effects.columns:
                binary_Effects.at[idx, col] = 1
    binary_Effects = binary_Effects.to_numpy()

 
    mask_comparing_two_halves = matrix_cross_compare(binary_S) # returns the "mask" of where (columns) the rows of binary_S contradict
    indices_of_contradictions_among_preconditions = np.where(mask_comparing_two_halves)[0] 

    indices_of_NON_contradictions_among_preconditions = np.where(mask_comparing_two_halves == False)[0]
    binary_S_for_cond_effects = binary_S.copy()  #  the llps BUT only with the literals that contradict each others !!!
    binary_S_for_cond_effects[:, indices_of_NON_contradictions_among_preconditions] = 0


    indices_of_contradictions_among_preconditions = np.where(mask_comparing_two_halves == True)[0]
    binary_S_only_NON_contradicting_literals = binary_S.copy()
    binary_S_only_NON_contradicting_literals[:, indices_of_contradictions_among_preconditions] = 0

    binary_S_only_NON_contradicting_literals_unique = np.unique(binary_S_only_NON_contradicting_literals, axis=0)

    general_or_part = []

    # if (np.all(binary_S_only_NON_contradicting_literals_unique == 0, axis=1)).any():
    #     pass
    #     #print("A ROW HAS ONLY ZEROS, GENERAL PRECOND OR PART CANCELLED !")
    #     #print("WHICH MEANS THAT SOME LLP WOULD BE ALWAYS ACCEPTED IF WE EVER BUILD A GENERAL PRECONDITION PART
    #     #  SUCH THAT IT IS USELESS TO CONSTRUCT ONE (eg we could havve  (OR something True) )")
    # else:
        
    #     general_or_part = fromBinaryMatrixToStrings(binary_S_only_NON_contradicting_literals_unique)
    #     #print(general_or_part)

    

    # if len(general_or_part) > 0:
    #     print("axction num {}".format(num_action))
    #     print(general_or_part)
    #     exit()






    ### RETRIEVE THE CLUSTERS (ids of llas) OF THE CURRENT HLA



    ####   Check if, for a same action, some clusters have exactly the same effects

    
    #with open("clusterings/"+str(num_action)+"_clusters_True_both.txt", 'r') as ff:
    with open(base_dir+"/clusterings/"+str(num_action)+"_clusters_"+str(args.clustering_with_penalty)+"_"+args.clustering_base_data+".txt", 'r') as ff:

        clusters = {}

        already_present_effects = []


        for ijij, line in enumerate(ff):

            #print("ijij is {}".format(str(ijij)))

            clusters[ijij] = {}
            arr = np.fromstring(line.strip(), sep=' ', dtype=int)        
            #print("ARR is {}".format(str(arr)))   
            clusters[ijij]["preconds"] = binary_S[arr]  # was binary_S_for_cond_effects[arr]

            precondss = fromBinaryMatrixToStrings(clusters[ijij]["preconds"])

            clusters[ijij]["effects"] = binary_Effects[arr]

            effectss = fromBinaryMatrixToStrings(clusters[ijij]["effects"])


            if effectss not in already_present_effects:
                already_present_effects.append(effectss)
            else: 
                print("EFFECTS already present !!!!!")
                exit()

        all_clusters[num_action] = clusters


    gen_gen_gen_whens_for_cluster = ""
    gen_gen_gen_precond = "(OR "



    ### FOR EACH CLUSTER, CONSTRUCT THE "CLUSTER" PDDL ACTION 

    for id_, clus in clusters.items():

        if num_action == 8 and id_ == 1:
            for precond in fromBinaryMatrixToStrings(clus["preconds"]):
                print(precond)
            print()
            for effect in fromBinaryMatrixToStrings(clus["effects"]):
                print(effect)


        # # print tout en z1 etc
        # # 1) print ttes les llas concernées
        # # 2) print la Precond
        # # 3) print l'effet


        ##### LOGICAL_EQUIVALENCE_TESTING FROM R-LATPLAN ##############

        paths_preconditions = [] # a list of sublists where each sublist is a list of atoms (preconditions)
        paths_effects = []

        # INTERSECTION
        intersection_in_preconds = np.all(clus["preconds"] == 1, axis=0).astype(np.uint8)
        literals_of_intersection = np.where(intersection_in_preconds)[0]





        # enleve l'intersection PUIS,  parcours ligne par ligne et récupère
        preconds_of_clus_minus_inter = clus["preconds"].copy()
        preconds_of_clus_minus_inter[:, literals_of_intersection] = 0


        ### BUILD THE "OR" part of the general precondition
        in_or = []
        for linee in preconds_of_clus_minus_inter:
            in_or_tmp = []
            indices_of_lits = np.where(linee)[0]
            for inddeexx in indices_of_lits:
                in_or_tmp.append(atoms[inddeexx])
            in_or.append(in_or_tmp)  
        or_str_of_gen_precond_for_clus = ""


        if len(in_or) > 0:
            or_str_of_gen_precond_for_clus += "(OR "
            for one_part in in_or:
                one_part_str = "(AND "
                for att in one_part:
                    one_part_str += att + " "
                one_part_str += ")"
                or_str_of_gen_precond_for_clus += one_part_str
            or_str_of_gen_precond_for_clus += ")"
        if or_str_of_gen_precond_for_clus == "(OR (AND ))":
            or_str_of_gen_precond_for_clus = ""


        if num_action == 8 and id_ == 1:
            print("transform_atoms_listtransform_atoms_list")
            #print(or_str_of_gen_precond_for_clus)
            for inor in transform_atoms_list(in_or):
                print(inor)
            #rint(format_literals_1(or_str_of_gen_precond_for_clus))
            exit()

        ### BUILD THE "AND" part (here, just the list of literals) of the general precondition
        literals_of_intersection_str_list = []
        for lii in literals_of_intersection:
            literals_of_intersection_str_list.append(atoms[lii])

        
        

        # if num_action == 8 and id_ == 1:
        #     print("literals_of_intersection_str_list")
        #     print(literals_of_intersection_str_list)

        ########## LOGICAL_EQUIVALENCE_TESTING ##########
        and_base_for_paths_preconditions = set(I + literals_of_intersection_str_list)
        if len(in_or) > 0:
            for or_path in in_or:
                paths_preconditions.append(set(or_path + list(and_base_for_paths_preconditions)))
        else:
            paths_preconditions.append(and_base_for_paths_preconditions)
        # Adding the paths of general preconditions (test1) into the very general dico (we test everything at the end)
        paths_preconditions_all_high_lvl_actions_all_clusters[num_action][id_] = paths_preconditions 
        ########## END LOGICAL_EQUIVALENCE_TESTING ##########


        ### BUILD VERY GENERAL INTERSECTION OF THE PRECONDITION 
        gen_gen_inter = ""
        for litt in I:
            gen_gen_inter += litt + " "
        
        ##### BUILD the OR part of the general precondition for the cluster (gen_gen_precond)
        gen_gen_or = ""
        if len(general_or_part) > 0: # general_or_part is very specific and in reality always empty
            gen_gen_or += "(OR "
            for ell in general_or_part:
                gen_gen_or += ell + " "
            gen_gen_or += ")"

        gen_gen_precond = "(AND "

        gen_gen_precond += gen_gen_inter
        gen_gen_precond += gen_gen_or

        for ell in literals_of_intersection_str_list:
            gen_gen_precond += ell + " "

        if or_str_of_gen_precond_for_clus != "":
            gen_gen_precond += or_str_of_gen_precond_for_clus

        gen_gen_precond += ")"


        #if id_ < 3:
            
        gen_gen_gen_precond += " " + gen_gen_precond + " "

        ### BUILD THE 'EFFECT' PART
        intersection_in_effects = np.all(clus["effects"] == 1, axis=0).astype(np.uint8) # [ 1 0 0 1 0 ... 0] 1D array showing which literals are in common among all the llas effects
        disjunction_in_effects = np.any(clus["effects"] == 1, axis=0).astype(np.uint8) # [ 1 0 0 1 0 ... 0] 1D array showing which literals are in the "disjunction" of the llas effects
        indices_of_inter_literals_in_effects = np.where(intersection_in_effects)[0] # indices of the intersection: [0 3]

        mask__ = np.zeros(clus["effects"].shape[1], dtype=int)
        mask__[indices_of_inter_literals_in_effects] = 1
        matching_rows_with_intersect_effects = np.all(clus["effects"] == mask__, axis=1)
        llas_indices_with_intersect_effects = np.where(matching_rows_with_intersect_effects)[0] # indices of the llas of the cluster for which the effects EXACTLY correspond to the intersection of the effects


        # list of effects literals in the intersection
        inter_literals_in_effects = []
        for innn in indices_of_inter_literals_in_effects:
            inter_literals_in_effects.append(atoms[innn])
        effects_and_path = inter_literals_in_effects

        
        indices_of_disjun_literals_in_effects = np.where(disjunction_in_effects)[0]
        # indices of effects outside the intersection
        indices_of_outside_inter_of_literals_in_effects = indices_of_disjun_literals_in_effects[~np.isin(indices_of_disjun_literals_in_effects, indices_of_inter_literals_in_effects)]
        
        
        ### "WHEN" clauses
        when_clauses = []
        when_clauses_for_stats = []

        if len(indices_of_outside_inter_of_literals_in_effects) > 0:

            # for each index of an effect (index_lit) that is OUTISDE the intersection of effects for this cluster, we basically build a WHEN  
            for thefindex, index_lit in enumerate(indices_of_outside_inter_of_literals_in_effects):
                #print("thefindex {}/{}".format(str(thefindex), str(len(indices_of_outside_inter_of_literals_in_effects))))


                tmp_when_clause_for_stats = []
                indices_clust_elems_for_this_outside_effect = np.where(clus["effects"][:, index_lit] == 1)[0] # indices of llas (in the clus) which effects "cover" the outside effect
                preconds_for_this_effect = clus["preconds"][indices_clust_elems_for_this_outside_effect] # retrieve the corresponding preconds of these llas


                
                intersection_in_when_preconds = np.all(preconds_for_this_effect == 1, axis=0).astype(np.uint8) # retrieve the "AND" of the preconds for this "when"
                indices_of_inter_in_when_preconds = np.where(intersection_in_when_preconds)[0] # retrieve the indices of the preconds that are in common in all the llas of the "when"

                ### "AND" in preconds of the WHEN
                literals_of_intersection_in_when_preconds_str_list = [] # hold all literals that are in the intersection of the preconds of the "when"
                if len(indices_of_inter_in_when_preconds) > 0:
                    for indic in indices_of_inter_in_when_preconds:
                        if atoms[indic] not in and_base_for_paths_preconditions: # (we dont add the literals already in the "AND" of the general precond)
                            literals_of_intersection_in_when_preconds_str_list.append(atoms[indic])

                ### "OR" in preconds of the WHEN
                disjunction_in_when_preconds = np.any(preconds_for_this_effect == 1, axis=0).astype(np.uint8)
                indices_of_disjun_in_when_preconds = np.where(disjunction_in_when_preconds)[0]
                indices_of_outside_inter_of_literals_in_when_preconds = indices_of_disjun_in_when_preconds[~np.isin(indices_of_disjun_in_when_preconds, indices_of_inter_in_when_preconds)]
                literals_of_extersection_in_when_preconds_str_list = []
                if len(indices_of_outside_inter_of_literals_in_when_preconds) > 0:
                    for indic in indices_of_outside_inter_of_literals_in_when_preconds:
                        if atoms[indic] not in and_base_for_paths_preconditions: # (we dont add the literals already in the "AND" of the general precond)
                            literals_of_extersection_in_when_preconds_str_list.append(atoms[indic])
                

                # ########## LOGICAL_EQUIVALENCE_TESTING ##########
                # and_part_preconds_paths_for_this_when = set(literals_of_intersection_in_when_preconds_str_list)
                # if len(literals_of_extersection_in_when_preconds_str_list) > 0:
                #     all_combs = list(all_unique_combinations(literals_of_extersection_in_when_preconds_str_list))
                #     preconds_paths_for_this_when = parallel_preconds_with_progress(
                #         all_combs, 
                #         list(and_part_preconds_paths_for_this_when)
                #     )
                # else:
                #     preconds_paths_for_this_when = []


                ########## END LOGICAL_EQUIVALENCE_TESTING ##########



                ######## literals_of_intersection_in_when_preconds_str_list
                ######## literals_of_extersection_in_when_preconds_str_list
                
                and_part_precond = ""
                if len(literals_of_intersection_in_when_preconds_str_list) > 0:
                    for ll in literals_of_intersection_in_when_preconds_str_list:
                        and_part_precond += ll + " "  

                or_part_precond = ""
                if len(literals_of_extersection_in_when_preconds_str_list) > 0:
                    or_part_precond += "(OR "
                    for ll in literals_of_extersection_in_when_preconds_str_list:
                        or_part_precond += ll + " "
                    
                    or_part_precond += ")"


                whole_condition_part = ""

                if and_part_precond != "":
                    whole_condition_part += "(AND "
                    whole_condition_part += and_part_precond
                    whole_condition_part += or_part_precond
                    whole_condition_part += ")"
                else:
                    whole_condition_part = or_part_precond

                when_clause = two_tabs_space + "(when "
                when_clause += whole_condition_part + "\n"



                when_clause += two_tabs_space
                when_clause += atoms[index_lit] + "\n"
                when_clause += two_tabs_space + ")\n"

                when_clauses.append(when_clause)

                # ########## LOGICAL_EQUIVALENCE_TESTING ##########
                # effects_of_this_when = set(inter_literals_in_effects + [atoms[index_lit]])
                # for preconds_path_for_this_when in preconds_paths_for_this_when:
                #     tmp_when_clause_for_stats.append({ "preconds": preconds_path_for_this_when, "effects": effects_of_this_when })


                #     print(len(indices_of_disjun_in_when_preconds))
                #     print(len(indices_of_inter_in_when_preconds))
                #     print("YYY99999999999999")
                #     # if len(indices_of_disjun_in_when_preconds) == 23:
                #     #     exit()
                # when_clauses_for_stats.append(tmp_when_clause_for_stats)
                # ########## END LOGICAL_EQUIVALENCE_TESTING ##########

        all_conjunctions_for_when_of_gen_effects = []
        all_conjunctions_for_when_of_gen_effects_str = ""

        # if there is at least one when clause
        # we build a new when clause for the general effects
        if len(when_clauses) > 0:

            # # ########## LOGICAL_EQUIVALENCE_TESTING ##########
            # for when_c_gen in when_clauses_for_stats:
            #     for when_c in when_c_gen:
            #         all_conjunctions_for_when_of_gen_effects.append(list(when_c["preconds"]))
            # # ########## END LOGICAL_EQUIVALENCE_TESTING ##########

            if len(clus["preconds"][llas_indices_with_intersect_effects]) > 0:

                #print(fromBinaryMatrixToStrings(clus["preconds"][llas_indices_with_intersect_effects]))
                and_effects_preconds_ = []
                for row__ in clus["preconds"][llas_indices_with_intersect_effects]:
                    selected_atoms_ = [atoms[i] for i, val in enumerate(row__) if val == 1]
                    and_effects_preconds_.append(selected_atoms_)


                # Flatten all atoms and count occurrences
                atom_counts = Counter(atom for sublist in and_effects_preconds_ for atom in set(sublist))
                num_lists = len(and_effects_preconds_)

                # Find atoms that appear in *every* sublist
                atoms_in_all = {atom for atom, count in atom_counts.items() if count == num_lists}

                # Filter each sublist to remove those atoms
                filtered_and_effects_preconds_ = [[atom for atom in sublist if atom not in atoms_in_all] for sublist in and_effects_preconds_]

                all_conjunctions_for_when_of_gen_effects.extend(filtered_and_effects_preconds_)


                # we remove any common atoms present among the preconds
                

            all_conjunctions_for_when_of_gen_effects_str += "(OR "
            for conju in all_conjunctions_for_when_of_gen_effects:
                tmp_or = " (AND "
                for at in conju: 
                    tmp_or += at + " "
                tmp_or += ") "
                all_conjunctions_for_when_of_gen_effects_str += tmp_or
            all_conjunctions_for_when_of_gen_effects_str += ")"


        paths_effects_all_high_lvl_actions_all_clusters[num_action][id_] = when_clauses_for_stats



        #################### PUTTING THE PIECES TOGETHER

        general_when_for_cluster = two_tabs_space +  "(when \n"
        general_when_for_cluster += two_tabs_space + two_tabs_space + gen_gen_precond + "\n"


        cluster_str = "(:action a"+str(num_action)+"_"+str(id_)+"\n"

        cluster_str += two_tabs_space + ":parameters ()\n"

        cluster_str += two_tabs_space + ":precondition "+gen_gen_precond+"\n"

        cluster_str += two_tabs_space + ":effect (AND \n"


        # for each cluster we convert it into a when clause with "effects"
        effect_for_general_when_for_cluster = ""


        if ( len(when_clauses) + len(inter_literals_in_effects) ) > 1:
            effect_for_general_when_for_cluster += two_tabs_space + "(AND \n"
            
        if str(args.specific_whens) == "False":
            all_conjunctions_for_when_of_gen_effects_str = ""



        if all_conjunctions_for_when_of_gen_effects_str != "":

            cluster_str += two_tabs_space + "(when " + all_conjunctions_for_when_of_gen_effects_str + "\n"
            effect_for_general_when_for_cluster += two_tabs_space + "(when " + all_conjunctions_for_when_of_gen_effects_str + "\n"
            
            cluster_str += two_tabs_space + "(AND "
            effect_for_general_when_for_cluster += two_tabs_space + "(AND "

            for inter_lit in inter_literals_in_effects:
                cluster_str += two_tabs_space + inter_lit + "\n"
                effect_for_general_when_for_cluster += two_tabs_space + inter_lit + "\n"

            cluster_str += two_tabs_space + ") "
            effect_for_general_when_for_cluster += two_tabs_space + ") "
        
            cluster_str += two_tabs_space + ") \n"
            effect_for_general_when_for_cluster += two_tabs_space + ") \n"

        else:


            
            for inter_lit in inter_literals_in_effects:
                cluster_str += two_tabs_space + inter_lit + "\n"
                effect_for_general_when_for_cluster += two_tabs_space + inter_lit + "\n"


        # nbre_llas = len(dico_highlvlid_lowlvlactions[num_action])

        # if len(when_clauses) > 0:

        #     print("nbre llas for action {}, WITH when_clauses is {}".format(str(num_action), str(nbre_llas)))

        #     if num_action == 17:
        #         print(when_clauses)
        #         exit()

        # continue

        for tmp_clause in when_clauses:
            cluster_str += tmp_clause
            effect_for_general_when_for_cluster += tmp_clause


        


        if ( len(when_clauses) + len(inter_literals_in_effects) ) > 1:
            effect_for_general_when_for_cluster += two_tabs_space + ") \n"


        cluster_str += two_tabs_space + ")\n"
        effect_for_general_when_for_cluster += two_tabs_space + ")\n"

        cluster_str += ")\n"

        #effect_for_general_when_for_cluster += ")\n"
        general_when_for_cluster += two_tabs_space + effect_for_general_when_for_cluster
        # print("cluster_strcluster_str")
        # print(cluster_str)

        """        print("ALLER LA ")
                print(cluster_str)
                print("general_when_for_clustergeneral_when_for_clustergeneral_when_for_cluster")
                print(general_when_for_cluster)
                if id_ == 5:
                    break
                continue """

        # print("general_when_for_clustergeneral_when_for_clustergeneral_when_for_clustergeneral_when_for_cluster")
        # print(general_when_for_cluster)
        #gen_gen_gen_whens_for_cluster += general_when_for_cluster + "\n"
        gen_gen_gen_whens_for_cluster += textwrap.dedent(general_when_for_cluster) + "\n"

        
        # print("cluster_strcluster_strcluster_strcluster_str")
        # print(cluster_str)
        # exit()

        whole_actions_str += cluster_str

        high_lvl_action_str_gen += cluster_str
        continue

    gen_gen_gen_precond += ")"


    # high_lvl_action_str = "(:action a"+str(num_action)+"\n"
    # high_lvl_action_str += two_tabs_space + ":parameters ()\n"
    # high_lvl_action_str += two_tabs_space + ":precondition "+gen_gen_gen_precond+"\n"
    # #high_lvl_action_str += two_tabs_space + ":precondition ()"+"\n"
    # #high_lvl_action_str += two_tabs_space + ":precondition "+general_or_and+"\n"
    # high_lvl_action_str += two_tabs_space + ":effect (AND \n"
    # high_lvl_action_str += two_tabs_space + two_tabs_space + gen_gen_gen_whens_for_cluster
    # high_lvl_action_str += two_tabs_space + ")\n"
    # high_lvl_action_str += ")\n"


    # high_lvl_action_str_gen += high_lvl_action_str + "\n"




###  LOGICAL_EQUIVALENCE_TESTING, THE ACTUAL TESTINGS
    
counter_passed_test1 = 0
counter_NOT_passed_test1 = 0

llas_tested = {}
# for each very high level action
for num_action in range(0, nber_hlas):

    print("ON EST LA {}".format(num_action))

    # we retrieve its corresponding llas
    llas_tested[num_action] = {}

    # For each LLA (of the current hla)
    for thecounter, (lowlvlkey, dico_vals) in enumerate(dico_highlvlid_lowlvlactions[num_action].items()):
        
        
        # 
        llas_tested[num_action][lowlvlkey] = {}
        
        # TEST 1: there should be only one precondition path (among all of all the clusters) that should be equal to llps

        llps = dico_vals["preconds"]

        # we gather, for the lla tested, information about if its llp is accepted in ANY other "cluster action"
        llas_tested[num_action][lowlvlkey]["test1"] = {}

        # we go through ALLL clusters (num_action_bis is a real num_action lvl, it's just that it's asociated with a particular cluster)
        for num_action_bis, clusters_of_the_action in all_clusters.items():

            llas_tested[num_action][lowlvlkey]["test1"][num_action_bis] = {}

            # for each cluster of a high lvl action clustering
            for id_clus, clus in clusters_of_the_action.items():

                nber_of_equal_sets = 0
                for set_ in paths_preconditions_all_high_lvl_actions_all_clusters[num_action_bis][id_clus]:
                    if set(llps) == set_:
                        nber_of_equal_sets += 1


                
                
                if nber_of_equal_sets == 1:
                    # test1 passed !!  TELLS FOR A GIVEN CLUSTER (id_clus)  THAT THE LLA (lowlvlkey) PASSED TEST 1
                    llas_tested[num_action][lowlvlkey]["test1"][num_action_bis][id_clus]  = "OK"
                else:
                    llas_tested[num_action][lowlvlkey]["test1"][num_action_bis][id_clus]  = "NOT"




    #### AFTER THE LOOP WE CHECK THE TESTING DICOS
    #### TEST 1 !!
    counter_of_OKs_of_good_high_lvl_action = []
    counter_of_OKs_NOT_good_high_lvl_action = [] #0
    for num_ac_bis, dicos in llas_tested[num_action][lowlvlkey]["test1"].items():
        
        for id_clus_, stri in dicos.items():

            if stri == "OK" and num_ac_bis == num_action:
                counter_of_OKs_of_good_high_lvl_action.append({"hla": num_ac_bis, "clus": id_clus_})

            elif stri == "OK" and num_ac_bis != num_action:
                counter_of_OKs_NOT_good_high_lvl_action.append({"hla": num_ac_bis, "clus": id_clus_})

    if len(counter_of_OKs_of_good_high_lvl_action) == 1 and len(counter_of_OKs_NOT_good_high_lvl_action) == 0:
        print("LLA {} of hla {} passed the TEST1 successfully!".format(str(lowlvlkey), str(num_action)))
        counter_passed_test1 += 1



#     else:
#         counter_NOT_passed_test1 += 1
#         print("LLA {} of hla {} DID NOT passed the TEST1 !!!!".format(str(lowlvlkey), str(num_action)))
#         # print("counter_of_OKs_of_good_high_lvl_action {}".format(str(counter_of_OKs_of_good_high_lvl_action)))
#         # print("counter_of_OKs_NOT_good_high_lvl_action (other hla in which the llp are 'valid') {}".format(str(counter_of_OKs_NOT_good_high_lvl_action)))
#         # print(format_literals_1(dico_highlvlid_lowlvlactions[num_action][lowlvlkey]["preconds"]))
#         # z1  z9  z10  z12  z15  ~z0  ~z2  ~z4  ~z7  ~z8  ~z11  ~z13
        


# print("#LLAs passed test1: {}".format(str(counter_passed_test1)))
# print("#LLAs NOT passed test1: {}".format(str(counter_NOT_passed_test1)))



exit()


###################
###################   IV. WRITE THE PDDL 
###################

str_for_whens = ""
if str(args.specific_whens) == "True":
    str_for_whens += "_speWhens"


name_pddl_file = "domainClustered_llas_"+str(args.clustering_with_penalty)+"_"+args.clustering_base_data+str_for_whens



with open(base_dir + "/" + name_pddl_file+".pddl", "w") as f:

    f.write("(define (domain latent)\n")
    f.write("(:requirements :negative-preconditions :strips :conditional-effects)\n")
    f.write("(:types\n")
    f.write(")\n")
    f.write("(:predicates\n")

    for i in range(latent_size):
        f.write("(z"+str(i)+" )\n")
    #feature_names = [f"precondition_{i}" for i in range(X.shape[1])]
    f.write(")\n")



    f.write(whole_actions_str)
    #f.write(high_lvl_action_str_gen)
    f.write("\n")
    f.write(")")


