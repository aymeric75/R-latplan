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

###############################################################
####################### UTILS FUNCTIONS and CLASSES ###########
###############################################################



N=25

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


def factorize_dict(input_dict):


    effects_for_all = []
    for keyy, listes in input_dict.items():
        if not listes:
            effects_for_all.append(keyy)

    list_to_keys = defaultdict(set)
    new_dict = defaultdict(list)
    
    # Reverse mapping from lists to keys
    for key, lists in input_dict.items():
        # if not lists:
        #     new_dict[(key,)] = []  # Preserve empty lists with single key as tuple
        for lst in lists:
            tuple_lst = tuple(sorted(lst))  # Convert list to tuple for immutability and sorting for consistency
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
    print("eff_name is ")
    print(eff_name)
    if eff_name.split('_')[0] == 'del':
        eff_name = "(not (z"+eff_name.split('_')[1]+"))"
    else:
        eff_name = "(z"+eff_name.split('_')[1]+")"
    return eff_name


def sort_by_samples(lst):
    # Extract the number from the last element of each sublist and sort
    sorted_lst = sorted(lst, key=lambda x: int(x[-1].split(': ')[1]), reverse=True)
    return sorted_lst

###############################################################
####################### END UTILS FUNCTIONS ###################
###############################################################



###############################################################
####################### BUILD THE X AND Y ####################
###############################################################


path_to_dataset = "/workspace/R-latplan/r_latplan_datasets/sokoban/sokoban_complete_clean_faultless_withoutTI_N25/data.p"

loaded_data = load_dataset(path_to_dataset) # load dataset for the specific experiment
train_set_no_dupp = loaded_data["train_set_no_dupp_processed"]


nb_high_lvl = len(train_set_no_dupp[0][2])




dico_keyTransiGroundTruth_valueHighGroundTruth = {}

dico_keyHighGroundTruth_valueLowGroundTruth = {}


for ii, ele in enumerate(train_set_no_dupp): # loop over the train set (without dupplicate) # AND group the transitions into their respective High Level Actions

    if np.argmax(ele[1]) not in dico_keyTransiGroundTruth_valueHighGroundTruth:
        dico_keyTransiGroundTruth_valueHighGroundTruth[str(np.argmax(ele[1]))] = str(np.argmax(ele[2]))


    if np.argmax(ele[2]) not in dico_keyHighGroundTruth_valueLowGroundTruth:
        dico_keyHighGroundTruth_valueLowGroundTruth[np.argmax(ele[2])] = []

    if np.argmax(ele[1]) not in dico_keyHighGroundTruth_valueLowGroundTruth[np.argmax(ele[2])]:
        dico_keyHighGroundTruth_valueLowGroundTruth[np.argmax(ele[2])].append(np.argmax(ele[1]))





################################################################
################# MANAGE THE XORS  ####################
##############################################################

### ie, to each "transition ID (of the csv), associate the high lvl action"

dico_keyTransiCSV_valueHigh = {}

domainfile="domain.pddl"
problemfile="problem.pddl"
translate_path = os.path.join("/workspace/R-latplan", "downward", "src", "translate")
sys.path.insert(0, translate_path)

dicoCsvInd_groundTruthHigh = {}

try:
    import FDgrounder
    from FDgrounder import pddl_parser as pddl_pars
    task = pddl_pars.open(
        domain_filename=domainfile, task_filename=problemfile) # options.task
    
    print(FDgrounder.__file__)



    nb_total_actions = len(task.actions) + len(task.indices_actions_no_effects) # including actions with no effects

    cc_normal_actions = 0

    print(task.names_actions_no_effects)
    #exit()

    for i in range(nb_total_actions):

            low_lvl_name = ""

            if i in task.indices_actions_no_effects:

                index_ac =  task.indices_actions_no_effects.index(i)

                low_lvl_name = task.names_actions_no_effects[index_ac]
            
            else:

                act = task.actions[cc_normal_actions]

                low_lvl_name = act.name

                cc_normal_actions += 1

            low_lvl_name_clean = low_lvl_name.split("-")[0].split("+")[0]

            highname = dico_keyTransiGroundTruth_valueHighGroundTruth[low_lvl_name_clean[1:]]

            dicoCsvInd_groundTruthHigh[i] = highname


finally:
    # Restore sys.path to avoid conflicts
    sys.path.pop(0)

print("dicoCsvInd_groundTruthHighdicoCsvInd_groundTruthHighdicoCsvInd_groundTruthHigh")
print(dicoCsvInd_groundTruthHigh)


# NOW USE THE dico_keyTransiCSV_valueHigh in order to have, for each CSV id, the corres high lvl


# FIRST create dico_transitions_per_high_lvl_actions 
dico_transitions_per_high_lvl_actions = {} # GROUP THE TRANSITIONS BY THEIR HIGH LVL ACTION
for ac in range(nb_high_lvl):

    for k,v in dicoCsvInd_groundTruthHigh.items():

        if str(v) == str(ac):

            if ac not in dico_transitions_per_high_lvl_actions:
                dico_transitions_per_high_lvl_actions[ac] = []
            
            dico_transitions_per_high_lvl_actions[ac].append(int(k))

print("dico_transitions_per_high_lvl_actionsdico_transitions_per_high_lvl_actionsdico_transitions_per_high_lvl_actionsdico_transitions_per_high_lvl_actions")


pos_preconds = np.loadtxt("pos_preconds_aligned.csv", delimiter=' ', dtype=int)
neg_preconds = np.loadtxt("neg_preconds_aligned.csv", delimiter=' ', dtype=int)

add_effs = np.loadtxt("add_effs_aligned.csv", delimiter=' ', dtype=int)
del_effs = np.loadtxt("del_effs_aligned.csv", delimiter=' ', dtype=int)

effects_set = []
for i in range(N):
    effects_set.append(f"add_{i}")
for i in range(N):
    effects_set.append(f"del_{i}")
preconds_perEff_perAction = {}

feature_names = []
for i in range(N):
    feature_names.append(f"(z_{i}_1)")
    feature_names.append(f"(z_{i}_0)")

preconds_names = []
for f in feature_names:
    preconds_names.append(format_precond(f.replace("(", "").replace(")", "")))



# ###############################################################
# ##############      X and Y FOR THE SECOND DT           #######
# ###############################################################


# all_transitions_preconds = np.zeros((len(pos_preconds), len(pos_preconds[0])*2)) # (1469, 32)
# all_Ys = np.zeros((len(pos_preconds), nb_high_lvl)) # (1469, 22)


# for num_action in range(0, nb_high_lvl):
#     for trans_id in dico_transitions_per_high_lvl_actions[num_action].keys():
#         all_Ys[trans_id][num_action] = 1
#         preconds_union = []
#         preconds_union.extend(pos_preconds[trans_id])
#         preconds_union.extend(neg_preconds[trans_id])
#         all_transitions_preconds[trans_id] += preconds_union

# #print(np.sum(np.sum(all_Ys, axis=1)))



# all_transitions_preconds = pd.DataFrame(all_transitions_preconds)

# new_columns = []
# for i in range(16):
#     new_columns.append(f"z_{i}")
# for i in range(16):
#     new_columns.append(f"not(z_{i})")

# all_transitions_preconds.columns = new_columns[:len(all_transitions_preconds.columns)] # Use slicing to handle cases where X might have fewer columns


# # print(all_transitions_preconds.head())
# # exit()
# new_entire_X = pd.DataFrame()
# for i in range(16):
#     # Create a new column in new_X
#     new_column = []
#     for index, row in all_transitions_preconds.iterrows():
#         if row[f'z_{i}'] == 1:
#             new_column.append(1)
#         elif row[f'not(z_{i})'] == 1:
#             new_column.append(0)
#         else:
#             new_column.append('?')  # Or any other representation you prefer for "otherwise"
#     new_entire_X[f'z_{i}'] = new_column  # Assign the new column to the new DataFrame



# new_entire_X.replace(1, "1", inplace=True)
# new_entire_X.replace(0, "0", inplace=True)

# onehot_encoder = OneHotEncoder(sparse=False, categories = [["1", "0", "?"] for i in range(16)])  # 'first' to drop the first category to avoid multicollinearity
# onehot_encoded = onehot_encoder.fit_transform(new_entire_X) # Fit and transform the data
# new_entire_X = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(['z_'+str(i) for i in range(16)]))

# # removing the "?" columns
# new_entire_X = new_entire_X.loc[:, ~new_entire_X.columns.str.contains(r'\?')]


# ###############################################################
# ################     end X and Y FOR THE NEW DT         #######
# ###############################################################




##############################################################################
################  LOOP OVER EACH ACTION and learn the First tree #############
##############################################################################

for num_action in range(0, nb_high_lvl):

    # if num_action != 23:
    #     continue

    preconds_perEff_perAction[num_action] = {}

    ##################################################################################
    # Construct the X
    action_transitions_preconds = []
    for trans_id in dico_transitions_per_high_lvl_actions[num_action]:
        preconds_union = []
        preconds_union.extend(pos_preconds[trans_id])
        preconds_union.extend(neg_preconds[trans_id])
        action_transitions_preconds.append(preconds_union)
    X = np.array(action_transitions_preconds)

    print("1111")
    print(X.shape)

    X = pd.DataFrame(X)
    new_columns = []
    for i in range(N):
        new_columns.append(f"z_{i}")
    for i in range(N):
        new_columns.append(f"not(z_{i})")
    X.columns = new_columns[:len(X.columns)] # Use slicing to handle cases where X might have fewer columns
    X.index = list(dico_transitions_per_high_lvl_actions[num_action])
    indices_with_nine = X[X.isin([9]).any(axis=1)].index
    X = X.drop(index=indices_with_nine)

    # Create a new DataFrame for the modified X matrix
    new_X = pd.DataFrame()
    for i in range(N):
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


    onehot_encoder = OneHotEncoder(sparse=False, categories = [["1", "0", "?"] for i in range(N)])  # 'first' to drop the first category to avoid multicollinearity
    onehot_encoded = onehot_encoder.fit_transform(new_X) # Fit and transform the data
    new_X = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(['z_'+str(i) for i in range(N)]))
    
    print("T LAAAA")
    print(new_X.head())
    #exit()
    # removing the "?" columns
    new_X = new_X.loc[:, ~new_X.columns.str.contains(r'\?')]


    # END Construct the X
    ##################################################################################



    ##################################################################################
    # Construct the Y
    action_transitions_effects = []
    for trans_id in dico_transitions_per_high_lvl_actions[num_action]:
        effects_union = []
        effects_union.extend(add_effs[trans_id])
        effects_union.extend(del_effs[trans_id])
        action_transitions_effects.append(effects_union)
    Y = np.array(action_transitions_effects)

    Y = pd.DataFrame(Y)
    new_columns = []
    for i in range(N):
        new_columns.append(f"add_{i}")
    for i in range(N):
        new_columns.append(f"del_{i}")
    Y.columns = new_columns[:len(Y.columns)] # Use slicing to handle cases where X might have fewer columns

    # if num_action == 23:
    #     print("Y HEAD")
    #     print(Y)
      
    #     exit()

    Y.index = list(dico_transitions_per_high_lvl_actions[num_action])
    Y = Y.drop(index=indices_with_nine)

    print(new_X.head())
    print()
    print(Y.head())
    print()

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

        # continue

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
                    for cond in pathh[:-1]:
                        if ">" in cond:
                            tmp.append(format_precond(cond.split(" > ")[0].replace("(", "").replace(")", "")))
                        else: # "<=" in cond
                            tmp.append(format_precond(cond.split(" <= ")[0].replace("(", "").replace(")", ""), reverse=True))
                            #tmp.append(format_precond(cond.split(" <= ")[0].replace("(", "").replace(")", "").replace("not", "")))
                            #break
                    if len(tmp) > 0:
                        beauty_paths.append(tmp)
            preconds_perEff_perAction[num_action][effect_name] = beauty_paths
            
        elif the_one_node_is_true:
            tmp = []
            beauty_paths.append(tmp)
            add_stuff = True
            preconds_perEff_perAction[num_action][effect_name] = []
        #print("=" * 50)  # Separator between trees


    # remove dupplicates from preconds groups
    for eff_n, list_lists in preconds_perEff_perAction[num_action].items():
        for ijijijij, l in enumerate(list_lists):
            l = list(set(l))
            preconds_perEff_perAction[num_action][eff_n][ijijijij] = l

    # remove contradictions from preconds groups
    for eff_n, list_lists in preconds_perEff_perAction[num_action].items():
        for ijijijij, l in enumerate(list_lists):
            l_ = l.copy()
            dones = []
            for ele in l:
                if opposite(ele) in dones:
                    l_.remove(ele)
                    l_.remove(opposite(ele))
                dones.append(ele)
            preconds_perEff_perAction[num_action][eff_n][ijijijij] = l_
    # END Print or/and return the DT branches (FIRST DT)
    ##################################################################################

##############################################################################
##############  END LOOP OVER EACH ACTION and learn the First tree   #########
##############################################################################



##############################################################################
###########################  WRITE THE PDDL   ################################
#######################  and LEARN THE SECOND TREE ###########################
##############################################################################
with open("domainCondBIS.pddl", "w") as f:


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

    for num_action in range(0, nb_high_lvl):


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


        merged_dict = factorize_dict(preconds_perEff_perAction[num_action])
        f.write("(:action a"+str(num_action)+"\n")
        f.write("   :parameters ()\n")

        f.write("   :precondition ()\n") # IF NOT PRECONDITIONS


        # new_entire_X_ = new_entire_X.copy()
        # liste_preconds_to_remove = []
        # liste_preconds_to_remove_bis = []


        # for prec in preconds_names:
        #     all_preconds_of_high_lvl = []
        #     for eff_group, preconds_sets in merged_dict.items():
        #         for prec_set in preconds_sets:
        #             for precc in prec_set:
        #                 all_preconds_of_high_lvl.append(precc)

        #             # if len(prec_set) == 1:

        #             #     if prec_set[0] == prec:

        #             #         liste_preconds_to_remove.append(prec)
        #             #         # exit()
        #             #         break


        #     if prec in all_preconds_of_high_lvl:
        #         liste_preconds_to_remove_bis.append(prec)
        #         #[['(not (z13))', '(not (z5))', '(not (z7))', '(z6)'], ['(not (z0))', ......

        # if len(liste_preconds_to_remove) > 0:
        #     new_entire_X_ = new_entire_X_.drop(columns=[reverse_format(prec_) for prec_ in liste_preconds_to_remove])

        # if not new_entire_X_.empty:

        #     # For each action, do the DT of preconds for knowning if apply the action or not

        #     clf2 = DecisionTreeClassifier(criterion='entropy', random_state=42)
        #     clf2.fit(new_entire_X_, all_Ys[:, num_action])

        #     ### Routine to count the # blue leaves
        #     # tree = clf2.tree_
        #     # # Dictionary to store count of leaves by number of samples
        #     # leaf_samples_dict = defaultdict(int)
        #     # for node in range(tree.node_count):
        #     #     if tree.children_left[node] == -1 and tree.children_right[node] == -1:  # Check if it's a leaf
        #     #         predicted_class = np.argmax(tree.value[node])  # Get the predicted class
        #     #         if predicted_class == 1:  # Only count leaves with class 1
        #     #             num_samples = int(tree.n_node_samples[node])  # Get the number of samples
        #     #             leaf_samples_dict[num_samples] += 1  # Increment count for that sample size
        #     ## Convert defaultdict to a regular dict
        #     # leaf_samples_dict = dict(leaf_samples_dict)
        #     # print("Leaves with class 1, grouped by number of samples:", leaf_samples_dict)



        #     ugly_paths = get_rules(clf2, feature_names, None)


            
        #     # plt.figure(figsize=(72, 72))
        #     # plot_tree(clf2, 
        #     #         feature_names=feature_names, 
        #     #         label= 'all',
        #     #         class_names=[f'Class {k}' for k in range(2)], 
        #     #         filled=True, 
        #     #         rounded=True,
        #     #         impurity = True
        #     #         )
        #     # say_one_positive_node = ""
        #     # plt.title(f"Decision Tree for Action {num_action}, {say_one_positive_node}", fontsize = 40)
        #     # file_name = f"decision_tree_HIGH_LEVEL_{num_action}.png"
        #     # plt.savefig(file_name, format="png", dpi=300)  

        #     # exit()


        #     one_node = False
        #     the_one_node_is_true = False
        #     add_stuff = False
        #     if len(ugly_paths) == 1 and len(ugly_paths[0]) == 1:
        #         one_node = True
        #         summ = int(all_Ys[:, num_action].sum()) # check if class is 1, ie is the sum is > 0
        #         # means that the effect was "1" (ie applied) at least once
        #         if summ > 0:
        #             ugly_paths[0][0].replace('0', '1')
        #             the_one_node_is_true = True
    
        #     beauty_paths = []
        #     if not one_node:
        #         add_stuff = True
        #         for pathh in ugly_paths:
        #             if 'class is: 0' in pathh[-1]:
        #                 continue
        #             elif 'class is: 1' in pathh[-1]: # and len(pathh[:-1]) > 2:
        #                 tmp = [] #{}
        #                 integer = int(''.join(x for x in str(pathh[-1].split(", ")[1]) if x.isdigit()))
        #                 #tmp[integer] = []
        #                 for cond in pathh[:-1]:
        #                     if ">" in cond:
        #                         tmp.append(format_precond(cond.split(" > ")[0].replace("(", "").replace(")", "")))
        #                     else: # "<=" in cond
        #                         tmp.append(format_precond(cond.split(" <= ")[0].replace("(", "").replace(")", ""), reverse=True))
        #                         #tmp.append(format_precond(cond.split(" <= ")[0].replace("(", "").replace(")", "").replace("not", "")))
        #                         #break
        #                 tmp.append("#samples: "+str(integer))
        #                 if len(tmp) > 0:
        #                     beauty_paths.append(tmp)

        #         preconds_perEff_perAction[num_action][effect_name] = beauty_paths

        #     elif the_one_node_is_true:
        #         tmp = []
        #         integer = int(''.join(x for x in str(ugly_paths[0].split(", ")[1]) if x.isdigit()))
        #         tmp.append("#samples: "+str(integer))
        #         beauty_paths.append(tmp)
        #         add_stuff = True
        #         preconds_perEff_perAction[num_action][effect_name] = []
        # else:
        #     print("BIG PROBLEM !!")
        #     exit()
        # beauty_paths_sorted = sort_by_samples(beauty_paths)



        # for bb in beauty_paths_sorted:
        #     if "#samples: 1" in bb[-1] or "#samples: 2" in bb[-1]:
        #         beauty_paths_sorted.remove(bb)

        # for ijijijij, l in enumerate(beauty_paths_sorted):
        #     l_ = l.copy()
        #     dones = []
        #     for ele in l[:-1]:
      
        #         if opposite(ele) in dones:
        #             l_.remove(ele)
        #             l_.remove(opposite(ele))
        #         dones.append(ele)

        #     beauty_paths_sorted[ijijijij] = l_

        # for ijijijij, l in enumerate(beauty_paths_sorted):
        #     l_ = list(set(l[:-1]))
        #     l_.append(l[-1])
        #     beauty_paths_sorted[ijijijij] = l_


        # # print("SIZE beauty_paths_sorted {}".format(str(len(beauty_paths_sorted))))
        # # continue


        # dico_disjunctions_preconds_dt[num_action] = beauty_paths_sorted

        # # CASE : precond is an OR of the set of all high lvl preconds 
        # f.write("   :precondition (OR ")
        # for liste in dico_disjunctions_preconds_dt[num_action]:
        #     tmp_str = "(AND "
        #     for ele in liste[:-1]:
        #         #ele = transfo_precond_effect_name_to_pddl(ele)
        #         tmp_str += " "+ele 
        #     tmp_str += ")"
        #     f.write(" "+tmp_str)
        # f.write(")\n") # closing the (


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
                print("eff_name {}".format(str(eff_name)))
                #tmp_group_of_effects += two_tabs_space+eff_name + "\n"
                str_preconds += two_tabs_space + eff_name + "\n"
            f.write(str_preconds)



        f.write("   )\n")
        f.write(")\n")

    
    f.write(")\n")

