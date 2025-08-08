# Algo that clusters low level actions of R-latplan
# construct clusterings for each high level action and 
# put the low level action ids for each cluster into a file (each high lvl action has a file)
# it tests and create clusters from different hyper params combinations based on
# to use a Penalty for the Jaccard distance or not, to cluster based on only the preconds or the effects or both

import numpy as np
import pandas as pd
import re
from collections import defaultdict
import os
import sys
from itertools import combinations, islice
import time
import itertools
from tqdm import tqdm
import pickle
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
import math
from typing import List
from math import comb
from itertools import islice
import argparse


parser = argparse.ArgumentParser(description="A script cluster low level actions of R-latplan")

parser.add_argument('--base_dir', default=None, type=str, help='Optional: Base path of the current experiment', required=False)
parser.add_argument('--data_folder', default=None, type=str, help='Optional: Base path of the current experiment data', required=False)


args = parser.parse_args()

base_dir = args.base_dir
data_folder = args.data_folder


if not os.path.exists(base_dir+"/clusterings"):
    os.makedirs(base_dir+"/clusterings")



two_tabs_space  = "         "



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





# return best clusters (indices), their number, the silouette score,
# THEN, afterwards with the indices, you need to compute the contractions in each cluster, (regarding the preconds and regarding the effects)
def clustering(data, penalized = True, lamb = 1.0):

    nber_elements = 0

    if type(data) is list and len(data) == 2:
        print("la")
        # common_ones
        # dist_matrix_1 = pdist(data[0], lambda u, v: common_ones(u, v))
        # dist_matrix_2 = pdist(data[1], lambda u, v: common_ones(u, v))
        dist_matrix_1 = pdist(data[0], lambda u, v: jaccard(u, v, penalized = penalized, lamb=lamb))
        dist_matrix_2 = pdist(data[1], lambda u, v: jaccard(u, v, penalized = penalized, lamb=lamb))
        dist_matrix = dist_matrix_1 + dist_matrix_2
        nber_elements = len(data[0])
    else:
        print("li")
        # Compute Jaccard distance matrix
        dist_matrix = pdist(data, lambda u, v: jaccard(u, v, penalized = penalized, lamb=lamb))
        #dist_matrix = pdist(data, lambda u, v: common_ones(u, v))
        nber_elements = len(data)
    


    # Perform hierarchical clustering
    Z = linkage(dist_matrix, method='average')

    # Try different cluster counts and compute silhouette scores
    best_score = -1
    best_k = None
    best_labels = None

    for k in range(2, nber_elements):

        labels = fcluster(Z, k, criterion='maxclust')
        score = silhouette_score(squareform(dist_matrix), labels, metric='precomputed')

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels


    # au final
    #
    #
    #    ce que tu veux, c'est:
    #
    #               on a testé plusieurs trucs
    #
    #                   voici les résultats des clustering
    #
    #                       ENSUTE TU COMMENTE LES RESULTATS
    #
    #
    #                       ET TU DIS QU ON A CHOISIT TELLE CLUSTERING
    #
    #
    #
    #
    #                       ET FINALEMENT TU DIS: COMMENT ON A FAIT LA TRANSLATION !!!!!
    #
    #
    #                           PUIS LES RESULTATS EN TERME DE ETC.
    #
    #
    #
    #                           

    # group_indices = defaultdict(list)
    # for idx, group_id in enumerate(best_labels):
    #     group_indices[group_id].append(idx)
    # list_of_indices_in_each_group = list(group_indices.values())

    
    # for r in list_of_indices_in_each_group:
    #     print("size group: {}, size of AND: {}".format(len(data[r]), np.sum(np.bitwise_and.reduce(data[r], axis=0))))

    # exit()

    # Organize results: group sample indices by cluster
    clusters = defaultdict(list)
    for idx, label in enumerate(best_labels):
        clusters[label].append(idx)


    return clusters, best_k, best_score




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



def group_identical_rows(arr):
    # Convert each row to a tuple so it can be used as a dictionary key
    row_dict = defaultdict(list)
    for idx, row in enumerate(arr):
        row_dict[tuple(row)].append(idx)
    
    return list(row_dict.values())





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

    print(FDgrounder.__file__)
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


            # print("act.name")
            # print(act.name)
            # print("low_lvl_name_cleanlow_lvl_name_clean")
            # print(low_lvl_name_clean)
            # print(low_lvl_name_clean[1:])
            # exit()
            # low_lvl_name_clean is the PDDL id  (from R-latplan)

            # highlvlid is the high lvl id

            # trans_id is just another id for low level actions, but 
            # which follows the numerical order from range 0 to #pddl actions


            highlvlid = dico_lowlvl_highlvl[int(low_lvl_name_clean[1:])]

            # print(highlvlid)
        

            if highlvlid not in dico_highlvlid_lowlvlactions:
                dico_highlvlid_lowlvlactions[highlvlid] = {}


            ##### CASE WHERE we DO the clusterings for then building the clustered PDDL 

            # if trans_id not in dico_highlvlid_lowlvlactions[highlvlid]:
            #     dico_highlvlid_lowlvlactions[highlvlid][trans_id] = {"preconds": [], "effects": []}


            # for precond in list(act.precondition.parts):
            #     f_name_precond = friendly_name(precond)
            #     dico_highlvlid_lowlvlactions[highlvlid][trans_id]["preconds"].append(f_name_precond)

            # for eff in act.effects:
            #     f_name_eff = friendly_name(eff.literal)
            #     dico_highlvlid_lowlvlactions[highlvlid][trans_id]["effects"].append(f_name_eff)


            ##### CASE WHERE we need the real ID of the low level actions
            if low_lvl_name_clean not in dico_highlvlid_lowlvlactions[highlvlid]:
                dico_highlvlid_lowlvlactions[highlvlid][low_lvl_name_clean] = {"preconds": [], "effects": []}


            for precond in list(act.precondition.parts):
                f_name_precond = friendly_name(precond)
                dico_highlvlid_lowlvlactions[highlvlid][low_lvl_name_clean]["preconds"].append(f_name_precond)

            for eff in act.effects:
                f_name_eff = friendly_name(eff.literal)
                dico_highlvlid_lowlvlactions[highlvlid][low_lvl_name_clean]["effects"].append(f_name_eff)



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

all_clusters = {}
all_clusters_preconds = {}


high_lvl_action_str_gen = ""

total_groups_count = 0


for num_action in range(0, nber_hlas):

    

    print("action is {}".format(str(num_action)))

    last_current_time = time.time()

    mapping_tmp_true = {}

    S = []
    E = [] 
    U_pre = []


    for thecounter, (lowlvlkey, dico_vals) in enumerate(dico_highlvlid_lowlvlactions[num_action].items()):
        mapping_tmp_true[thecounter] = lowlvlkey
        S.append(dico_vals["preconds"])
        E.append(dico_vals["effects"])
        for atom in dico_vals["preconds"]:
            U_pre.append(atom)


    U_pre = list(set(U_pre))

    # 1 -- taking the intersection of all the preconditions set of the high lvl action
    I = intersection_of_lists(S)


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



    ############  CLUSTERING ALGORITHM (it's a search, to comment)

    # data used  -
    dico__ = {

        # "only_preconds" : binary_S,
        # "only_effects" : binary_Effects,
        # "both" : [binary_S, binary_Effects],
        "by_same_effects": binary_Effects
    }


    columns = [ "       ", "only_preconds", "only_effects", "both", "by_same_effects"]

    rows = []

    #for penal in (True, False):
    
    for penal in (True,):

        row = []

        row.append(str(penal))

        for kkk, vvv in dico__.items():


            if kkk == "by_same_effects":

                # in this case, 
                groups = group_identical_rows(vvv)
   
                print("groups")
                print(groups)

                real_groups = []
                for g in groups:
                    real_groups.append([])
                    for ellll in g:
                        real_groups[-1].append(int(mapping_tmp_true[ellll][1:]))
                    


                total_groups_count += len(groups)

                # donc

                # 
                

                with open(base_dir+"/clusterings/"+str(num_action)+"_clusters_by_same_effects.txt", 'w') as fff1:
                    #for value in groups:
                    for value in real_groups:
                        fff1.write(' '.join(map(str, value)) + '\n')

            else:


                clusters_, best_k_, best_score_ = clustering(vvv, penalized = penal, lamb = 1. )


                total_contradictory_preconds = 0
                total_contradictory_effects = 0

                total_nb_lit_preconds = 0
                total_nb_lit_effects = 0

                for cluster_id, members in sorted(clusters_.items()):
                    

                    mask_comparing_two_halves_preconds = matrix_cross_compare(binary_S_for_cond_effects[members])
                    mask_comparing_two_halves_effects = matrix_cross_compare(binary_Effects[members])

                    nb_lit_preconds = len(np.where(sum_over_ax0(binary_S_for_cond_effects[members]))[0])
                    nb_lit_effects = len(np.where(sum_over_ax0(binary_Effects[members]))[0])

                    total_nb_lit_preconds += nb_lit_preconds
                    total_nb_lit_effects += nb_lit_effects

                    # pour chaque cluster, imprimer (eg z*) le nbre d'actions, les literals non contradictoires, les literaux contradictoires

                    indices_lit_contr_preconds = np.where(mask_comparing_two_halves_preconds)[0]
                    indices_lit_NON_contr_preconds = np.where(mask_comparing_two_halves_preconds == False)[0]

                    indices_lit_contr_effects = np.where(mask_comparing_two_halves_effects)[0]
                    indices_lit_NON_contr_effects = np.where(mask_comparing_two_halves_effects == False)[0]


                    #### la distance: sinon tu la base "uniquement" sur les contrictions, ie, le plus il y a d'atoms contradictoire le + grand la distance

                    total_contradictory_preconds += len(indices_lit_contr_preconds)
                    total_contradictory_effects += len(indices_lit_contr_effects)



                with open(base_dir+"/clusterings/"+str(num_action)+"_clusters_"+str(penal)+"_"+str(kkk)+".txt", 'w') as fff:
                    for value in clusters_.values():
                        fff.write(' '.join(map(str, value)) + '\n')

                
                mean_contr_preconds = round(total_contradictory_preconds / len(clusters_), 2)
                mean_contr_effects = round(total_contradictory_effects / len(clusters_), 2)
                mean_nb_lit_preconds = math.ceil(total_nb_lit_preconds / len(clusters_))
                mean_nb_lit_effects = math.ceil(total_nb_lit_effects / len(clusters_))

                print("penalty: {}, type: {}, nber clusters: {}, silouette: {}, mean_contr_preconds: {} / {}, mean_contr_effects: {} / {}, ".format(str(penal), kkk, best_k_, round(best_score_, 2), mean_contr_preconds, mean_nb_lit_preconds, mean_contr_effects, mean_nb_lit_effects))


                text_ = "#cluster: {}\nsilhouette: {}\n#con_pre: {}\n#con_eff: {}".format(str(best_k_), str(round(best_score_, 2)), str(mean_contr_preconds), str(mean_contr_effects))

                row.append(text_)
        
        rows.append(row)
        #print(rows)

    # from tabulate import tabulate

    # # Print the table
    # print(tabulate(rows, headers=columns, tablefmt="grid"))
    # exit()




print("total number of groups {}".format(total_groups_count))