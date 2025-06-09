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


base_dir = "/workspace/R-latplan/r_latplan_exps/hanoi/hanoi_complete_clean_faultless_withoutTI/"



two_tabs_space  = "         "



####    pour la precond générale TU VAS FAIRE CE QUE TU VOULAIS FAIRE à la base, C EST à DIR

#########         MMMMH, j'ai encore mieux, tu fais un clustering qui préviligie le clustering des preconds (elles se ressemblent le plus)

###################### OUAIS c'est beaucoup plus SIMPLE



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
            



def intersection_of_lists(lists: List[List[str]]) -> List[str]:
    if not lists:
        return []
    # Start with the first list as the base set
    common_elements = set(lists[0])
    # Intersect with each subsequent list
    for lst in lists[1:]:
        common_elements.intersection_update(lst)
    return list(common_elements)



def combinations_slice(args):
    """Worker function to generate a slice of combinations."""
    indices_U, r, start, end = args
    return list(islice(combinations(indices_U, r), start, end))


def GenerateCombinations(indices_U, maxSize):
    n = len(indices_U)
    total_combos = comb(n, maxSize)  # Total number of combinations
    max_workers = 64
    num_workers = min(cpu_count(), max_workers)
    chunk_size = total_combos // num_workers
    tasks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = total_combos if i == num_workers - 1 else (i + 1) * chunk_size
        tasks.append((indices_U, maxSize, start, end))
    results = []
    with Pool(processes=num_workers) as pool:
        for partial in tqdm(pool.imap_unordered(combinations_slice, tasks),
                            total=len(tasks),
                            desc="Generating combinations"):
            results.extend(partial)
    return results


def _check_entailment(args):
    candidate_idx, candidate, targets = args
    candidate = candidate.astype(bool)
    comparison = (targets & candidate) == candidate
    entailed = np.all(comparison, axis=1) # mask array of size #targets, where True at index i means target i entails candidate
    count = np.sum(entailed) # count the number of True values
    return candidate_idx, count

# returns a mask array of size #candidates, where True at index i means candidate i is equal to ele
def _check_final_set(args):
    ele, candidates = args
    matrix_ele = np.tile(ele, (candidates.shape[0], 1)).astype(bool)
    temp = np.all(matrix_ele == candidates, axis=1)
    return temp



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


def fromBinaryMatrixToStringsPDDL(sorted_combinationsMatrix_int):

    combinations_list = []
    for row in sorted_combinationsMatrix_int:
        selected_atoms = [atoms[i] for i, val in enumerate(row) if val == 1]
        combinations_list.append(selected_atoms)
    #print(combinations_list)

    
    #combinations_list_tranfo = transform_atoms_list(combinations_list)
    # for ele in combinations_list_tranfo:
    #     print(ele)
    
    return combinations_list



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

def find_compatible_groups(binary_S, n):
    num_items = binary_S.shape[0]
    binary_S_swapped = np.hstack((binary_S[:, n:], binary_S[:, :n]))

    def is_compatible(i, j):
        if i == j:
            return False
        return np.bitwise_and(binary_S[i], binary_S_swapped[j]).sum() == 0

    items = list(range(num_items))
    results = []
    bron_kerbosch([], items, [], is_compatible, results)
    return results


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


def jaccard(u, v, penalized = False, lamb = 1.):


    # 
    # u1 et u2
    # v1 et v2

    # et tu fais la jacard pour chaque PUIS l'addition


    m = len(u) // 2  # split point

    # Base intersection and union
    intersection = np.sum(np.logical_and(u, v))
    union = np.sum(np.logical_or(u, v))

    # Cross-half overlap penalty
    penalty_1 = np.sum(np.logical_and(u[:m], v[m:]))  # u's first half vs v's second half
    penalty_2 = np.sum(np.logical_and(u[m:], v[:m]))  # u's second half vs v's first half
    penalty = penalty_1 + penalty_2


    if union + penalty == 0:
        return 0.0  # define distance as 0 if completely zero vectors


    if penalized:
        #return 1 - (intersection / (union + penalty)) #+ lamb * (penalty / len(u))
        return 1 - (intersection / union) + lamb * (penalty / len(u))

    else:
        return 1 - (intersection / union)






def sum_over_ax0(thearray):
    sum_ = np.sum(thearray, axis=0)
    return sum_




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


# # return best clusters (indices), their number, the silouette score,
# # THEN, afterwards with the indices, you need to compute the contractions in each cluster, (regarding the preconds and regarding the effects)
# def clustering(data, penalized = True, lamb = 1.0):

#     nber_elements = 0

#     if type(data) is list and len(data) == 2:
#         print("la")
#         # common_ones
#         dist_matrix_1 = pdist(data[0], lambda u, v: common_ones(u, v))
#         dist_matrix_2 = pdist(data[1], lambda u, v: common_ones(u, v))
#         # dist_matrix_1 = pdist(data[0], lambda u, v: jaccard(u, v, penalized = penalized, lamb=lamb))
#         # dist_matrix_2 = pdist(data[1], lambda u, v: jaccard(u, v, penalized = penalized, lamb=lamb))
#         dist_matrix = dist_matrix_1 + dist_matrix_2
#         nber_elements = len(data[0])
#     else:
#         print("li")
#         # Compute Jaccard distance matrix
#         #dist_matrix = pdist(data, lambda u, v: jaccard(u, v, penalized = penalized, lamb=lamb))
#         dist_matrix = pdist(data, lambda u, v: common_ones(u, v))
#         nber_elements = len(data)
    


#     # Perform hierarchical clustering
#     Z = linkage(dist_matrix, method='average')

#     # Try different cluster counts and compute silhouette scores
#     best_score = -1
#     best_k = None
#     best_labels = None

#     for k in range(2, 10):

#         labels = fcluster(Z, k, criterion='maxclust')
#         score = silhouette_score(squareform(dist_matrix), labels, metric='precomputed')

#         if score > best_score:
#             best_score = score
#             best_k = k
#             best_labels = labels



#     group_indices = defaultdict(list)
#     for idx, group_id in enumerate(best_labels):
#         group_indices[group_id].append(idx)
#     list_of_indices_in_each_group = list(group_indices.values())

    
#     for r in list_of_indices_in_each_group:
#         print("size group: {}, size of AND: {}".format(len(data[r]), np.sum(np.bitwise_and.reduce(data[r], axis=0))))

#     exit()

#     # Organize results: group sample indices by cluster
#     clusters = defaultdict(list)
#     for idx, label in enumerate(best_labels):
#         clusters[label].append(idx)

    
#     # for each cluster, you cant to have the AND

#     # id de chaque merde


#     return clusters, best_k, best_score
def clustering(data, min_and_threshold=5):
    n = len(data)
    unassigned = set(range(n))
    groups = []

    while unassigned:
        seed_idx = unassigned.pop()
        group = [seed_idx]
        group_and = data[seed_idx].copy()

        to_remove = set()
        for idx in unassigned:
            candidate_and = np.bitwise_and(group_and, data[idx])
            if np.sum(candidate_and) >= min_and_threshold:
                group.append(idx)
                group_and = candidate_and
                to_remove.add(idx)

        unassigned -= to_remove
        groups.append(group)

    # print(len(groups))
    # # Print group info

    # for group in groups:
    #     print(f"size group: {len(group)}, size of AND: {np.sum(np.bitwise_and.reduce(data[group], axis=0))}")
    # exit()
    return groups




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



def remove_contradictions(literals):
    cleaned = set(literals)
    to_remove = set()

    for literal in cleaned:
        if literal.startswith('(not '):
            negated = literal[5:-1]  # removes '(not ' and ')'
        else:
            negated = f'(not {literal})'
        
        if negated in cleaned:
            to_remove.add(literal)
            to_remove.add(negated)

    return list(cleaned - to_remove)

###################
###################   I. CONSTRUCT THE DICO OF HIGH LVL VS LOW LVL ACTIONS  (use the domain.pddl)
###################

path_to_dataset = "/workspace/R-latplan/r_latplan_datasets/hanoi/hanoi_complete_clean_faultless_withoutTI/data.p"
loaded_data = load_dataset(path_to_dataset)
train_set_no_dupp = loaded_data["train_set_no_dupp_processed"]
dico_lowlvl_highlvl = {} 
for ii, ele in enumerate(train_set_no_dupp):
    if np.argmax(ele[1]) not in dico_lowlvl_highlvl:
        dico_lowlvl_highlvl[np.argmax(ele[1])] = np.argmax(ele[2])


dico_highlvlid_lowlvlactions = {}

#domainfile="domain_ORIGINAL_NO_DUPP.pddl"
domainfile="domain.pddl"
problemfile="problem.pddl"
translate_path = os.path.join("/workspace/R-latplan", "downward", "src", "translate")
sys.path.insert(0, translate_path)

try:
    import FDgrounder
    from FDgrounder import pddl_parser as pddl_pars
    task = pddl_pars.open(
    domain_filename=domainfile, task_filename=problemfile) 
    nb_total_actions = len(task.actions) + len(task.indices_actions_no_effects)
    cc_normal_actions = 0
    for trans_id in range(nb_total_actions):
        if trans_id in task.indices_actions_no_effects:
            pass 
        else:
            act = task.actions[cc_normal_actions]
            highlvlid = dico_lowlvl_highlvl[trans_id]
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



###################
###################   II. PREPARE THE STRING TO WRITE INTO THE PDDL FILE (whole_actions_str)
###################




atoms = [] # = U
for i in range(16):
    atoms.append(f"(z{i})")
for i in range(16):
    atoms.append(f"(not (z{i}))")


whole_actions_str = ""
paths_preconditions_all_high_lvl_actions_all_clusters = {}
paths_effects_all_high_lvl_actions_all_clusters = {}
all_clusters = {}
all_clusters_preconds = {}

veryhigh_lvl_action_str_gen = ""

for num_action in range(0, 22):

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

    binary_S = np.zeros((len(S), 32), dtype=np.uint8)
    binary_S = pd.DataFrame(binary_S)
    binary_S.columns = atoms
    for idx, row in binary_S.iterrows():
        for col in S[idx]:
            if col in binary_S.columns:
                binary_S.at[idx, col] = 1
    binary_S = binary_S.to_numpy()

    binary_Effects = np.zeros((len(E), 32), dtype=np.uint8)
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

    if (np.all(binary_S_only_NON_contradicting_literals_unique == 0, axis=1)).any():
        pass
        #print("A ROW HAS ONLY ZEROS, GENERAL PRECOND OR PART CANCELLED !")
        #print("WHICH MEANS THAT SOME LLP WOULD BE ALWAYS ACCEPTED IF WE EVER BUILD A GENERAL PRECONDITION PART
        #  SUCH THAT IT IS USELESS TO CONSTRUCT ONE (eg we could havve  (OR something True) )")
    else:
        
        general_or_part = fromBinaryMatrixToStringsPDDL(binary_S_only_NON_contradicting_literals_unique)
        # print(general_or_part) # [['(not (z6))', '(not (z14))'], ['(z5)'], ['(z5)', '(not (z14))'], ['(z5)', '(not (z6))'], ['(z5)', '(not (z6))', '(not (z14))'], ['(z1)', '(not (z14))'], ['(z1)', '(not (z6))', '(not (z14))'], ['(z1)', '(z5)', '(not (z14))'], ['(z1)', '(z5)', '(not (z6))'], ['(z1)', '(z5)', '(not (z6))', '(not (z14))']]
        # exit()
    

    # if len(general_or_part) > 0:
    #     print("axction num {}".format(num_action))
    #     print(general_or_part)
    #     exit()











    ###########  CLUSTERING ALGORITHM (it's a search, to comment)

    # data used  -
    dico__ = {

        "only_preconds" : binary_S, #binary_S_for_cond_effects, # 
        #"only_effects" : binary_Effects,
        #"both" : [binary_S_for_cond_effects, binary_Effects]
    }

    results = []

    for threshold in range(2, 10):

    
        for kkk, vvv in dico__.items():

            #clusters_, best_k_, best_score_ = clustering(vvv, penalized = penal, lamb = lamb_ )
            groups = clustering(vvv, min_and_threshold=threshold)

            avg_and = np.mean([np.sum(np.bitwise_and.reduce(vvv[g], axis=0)) for g in groups])
            results.append((threshold, len(groups), avg_and, groups))

            # with open("clusterings/"+str(num_action)+"_clusters_"+str(penal)+"_"+str(kkk)+".txt", 'w') as fff:
            #     for value in clusters_.values():
            #         fff.write(' '.join(map(str, value)) + '\n')

    #######  bon, des du 124 mais ya du 

    #########   IN ANY CASE, it has to be below a quarter of the size, and an AND above 3 (strictly)
            
    ########                THEN YOU TAKE THE BEST (lowest),   Groups/AND ratio 


    to_keep = []    

    for t, n, a, grps in results:
        #print(f"Threshold={t}: Groups={n}, Avg AND={a:.2f}")

        if n < len(binary_S) // 2 and a > 3:
            to_keep.append([t, n, a, grps])

    # print("to_keep")
    # print(to_keep)
    # exit()
    # if len(to_keep) == 0:
    #     print('probleme')
    #     exit()


    final_choice = None
    ratio = 999999
    for too in to_keep:
        n_ = too[1]
        a_ = too[2]
        ratio_tmp = n_/a_
        if ratio_tmp < ratio:
            ratio = ratio_tmp
            final_choice = too

    #print("action: {}, final_choice is with #size group: {}, avg ANDs: {}".format(str(num_action), final_choice[1], final_choice[2]))

    # if len(final_choice[-1]) == 0:
    #     print("THE FIN0")
    #     exit()
    # continue

    # Gonnna construct the freaking (OR (AND ) (AND )) clause as a general precond for the action !!!!!!!!!

    #print(final_choice[-1])

    print("STARTING for action {}".format(str(num_action)))
    list_of_conj = []

    binary_vecs_of_general_or_and = []

    for ands_ in final_choice[-1]:


        #print(binary_S[ands_])
        and_vector = np.bitwise_and.reduce(binary_S[ands_], axis=0)
        
        binary_vecs_of_general_or_and.append(and_vector)


        literalss = []
        indices = np.where(and_vector)[0]
        for iiinnn in indices:
            literalss.append(atoms[iiinnn])


        # print("literalss")
        # print(literalss)
        # exit()

        literalss_no_contradictions = remove_contradictions(literalss)

        if len(literalss_no_contradictions) != len(literalss):
            print("THERE WAS SOME COTNRADICTION !!!!")
            exit()
        list_of_conj.append(literalss_no_contradictions)

    
    general_or_and = "(OR "

    nber_of_ands = 0

    for conj in list_of_conj:
        tmp_and = " (AND "
        nber_of_ands += 1
        for c in conj:
            tmp_and += c + " "

        tmp_and += ") "

        general_or_and += tmp_and

    general_or_and += ")"

    # print("general_or_and")
    # print(general_or_and)
    print("nber_of_ands {}".format(str(nber_of_ands)))
    #continue

    #  what fucking ever

    ####   then... tu group contruis pour chaque action, le putain de AND
    

    #print(final_choice)




    # ###########################################################################################################################
    # ###########################################    CONSTRUCTING, A "WHEN" FOR EACH HIGH LEVEL EFFECT #########################


    # #   1) retrieve the indices of the llas where the effect is present

    # #   2) retrieve the corresponding llps

    # #   2 bis) output the nyumber of llps (of llas) for each effect

    # #   3) form a (AND (OR (AND ) (AND ) ... (AND )))  or a (OR (AND .. ) (AND ... ) (AND ...))
    
    # # generale_when_clause = ""
    
    # # nber_of_effects_to_use = 0

    # # for eff_ind in range(32):

    #     #print(binary_Effects.shape)
    #     # binary_Effects of shape (202, 32)
    #     binb = binary_Effects[binary_Effects[:, eff_ind] ==  1]

    #     indices_where_effect_present = np.where(binary_Effects[:, eff_ind])[0]

    #     #print(binb.shape) # (32, 32)

    #     cooresponding_llps = binary_S[indices_where_effect_present]

    #     # print("cooresponding_llps BEFORE")
    #     # print(cooresponding_llps)

    #     for theindexx, corres in enumerate(cooresponding_llps):

    #         for binary_vec in binary_vecs_of_general_or_and:

    #             if np.all((corres | binary_vec) == corres): # if corres includes binary_vec
    #                 cooresponding_llps[theindexx] -= binary_vec # then we substract binary_vec from corres



    #     # REMOVING LINES WHERE THERE ARE ONLY ZEROS
    #     cooresponding_llps = cooresponding_llps[~np.all(cooresponding_llps == 0, axis=1)]

    #     # REMOVING DUPLICATES (just leave the "unique" lines)
    #     cooresponding_llps = np.unique(cooresponding_llps, axis=0)



    #     indices_of_columns_with_only_ones = np.where(np.all(cooresponding_llps == 1, axis=0))[0]
        
    #     # if len(indices_of_columns_with_only_ones) > 0:

    #     #     print("cooresponding_llps AFTER")
    #     #     print(cooresponding_llps)

    #     #     print("indices_of_columns_with_only_ones")
    #     #     print(indices_of_columns_with_only_ones)
    #     #     exit()


    #     when_clause = ""

    #     condition_for_when_clause = ""

    #     if len(cooresponding_llps) > 0:
                


    #         if len(indices_of_columns_with_only_ones) > 0:

    #             condition_for_when_clause += "(AND "

    #             for indddddd in indices_of_columns_with_only_ones:

    #                 condition_for_when_clause += atoms[indddddd] + " "
                    
                


    #         # 

    #         or_part = "(OR "

    #         for cor_llp in cooresponding_llps:

    #             tmp_and_ = ""
    #             indices_literals_ = np.where(cor_llp)[0]
    #             if len(indices_literals_) > 0:
    #                 tmp_and_ += "(AND "
    #                 for iiiiiiiiiiiiii in indices_literals_:
    #                     tmp_and_ += atoms[iiiiiiiiiiiiii] + " "

    #                 tmp_and_ += ")"

    #             or_part += tmp_and_

    #         or_part += " ) "
            
    #         condition_for_when_clause += or_part


    #         if len(indices_of_columns_with_only_ones) > 0:
    #             condition_for_when_clause += " )"


        
    #     if len(cooresponding_llps) > 0 and condition_for_when_clause == "":
    #         print("YA PROBLEM LA !!!!!!!!")
    #         exit()

    #     elif len(cooresponding_llps) > 0 and condition_for_when_clause != "":
            
    #         when_clause += two_tabs_space + "(when \n"
    #         when_clause += two_tabs_space + condition_for_when_clause + "\n"
    #         when_clause += two_tabs_space + atoms[eff_ind] + "\n"
    #         when_clause += two_tabs_space + ")\n"

   
    #         generale_when_clause += when_clause


    #     if len(cooresponding_llps) > 0:
    #         nber_of_effects_to_use += 1

    # print("nber of effects {}".format(nber_of_effects_to_use))

    ##########################################################################################
    ##########################################################################################



    ### RETRIEVE THE CLUSTERS (ids of llas) OF THE CURRENT HLA

    #with open("clusterings/"+str(num_action)+"_clusters_True_both.txt", 'r') as ff:
    with open("clusterings/"+str(num_action)+"_clusters_True_only_effects.txt", 'r') as ff:

        clusters = {}

        for ijij, line in enumerate(ff):

            clusters[ijij] = {}
            arr = np.fromstring(line.strip(), sep=' ', dtype=int)           
            clusters[ijij]["preconds"] = binary_S[arr]  # was binary_S_for_cond_effects[arr]

            precondss = fromBinaryMatrixToStrings(clusters[ijij]["preconds"])

            clusters[ijij]["effects"] = binary_Effects[arr]
            effectss = fromBinaryMatrixToStrings(clusters[ijij]["effects"])

        all_clusters[num_action] = clusters


    gen_gen_gen_whens_for_cluster = ""
    gen_gen_gen_precond = "(OR "

    ### FOR EACH CLUSTER, CONSTRUCT THE "CLUSTER" PDDL ACTION 

    for id_, clus in clusters.items():


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



        ### BUILD THE "AND" part (here, just the list of literals) of the general precondition
        literals_of_intersection_str_list = []
        for lii in literals_of_intersection:
            literals_of_intersection_str_list.append(atoms[lii])




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
            for sublist in general_or_part:
                for ell in sublist:
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

                when_clause = ""
                if whole_condition_part != "":
                    when_clause += two_tabs_space + "(when "

                    when_clause += whole_condition_part + "\n"



                when_clause += two_tabs_space
                when_clause += atoms[index_lit] + "\n"

                if whole_condition_part != "":
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



        gen_gen_gen_whens_for_cluster += textwrap.dedent(general_when_for_cluster) + "\n"

























    action_str = "(:action a"+str(num_action)+"\n"

    action_str += two_tabs_space + ":parameters ()\n"

    action_str += two_tabs_space + ":precondition "+general_or_and+"\n"

    action_str += two_tabs_space + ":effect (AND \n"

    action_str += two_tabs_space +  gen_gen_gen_whens_for_cluster + "\n"

    action_str += two_tabs_space + ")\n"

    action_str +=  ")\n"
 
    veryhigh_lvl_action_str_gen += action_str



###################
###################   IV. WRITE THE PDDL 
###################


#name_pddl_file = "THENEWDOMAINBIS"
#name_pddl_file = "domainGenEffsWithWhens"
#name_pddl_file = "domainGenEffsWithoutWhens"
name_pddl_file = "domainClustered_VeryHigh"

with open(name_pddl_file+".pddl", "w") as f:

    f.write("(define (domain latent)\n")
    f.write("(:requirements :negative-preconditions :strips :conditional-effects)\n")
    f.write("(:types\n")
    f.write(")\n")
    f.write("(:predicates\n")

    for i in range(16):
        f.write("(z"+str(i)+" )\n")
    #feature_names = [f"precondition_{i}" for i in range(X.shape[1])]
    f.write(")\n")



    #f.write(whole_actions_str)
    f.write(veryhigh_lvl_action_str_gen)
    f.write("\n")
    f.write(")")


