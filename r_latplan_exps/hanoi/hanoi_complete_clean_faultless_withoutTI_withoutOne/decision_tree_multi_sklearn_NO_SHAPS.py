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

def load_dataset(path_to_file):
    import pickle
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    # print("path_to_file path_to_file")
    # print(path_to_file)
    # exit()
    return loaded_data



path_to_dataset = "/workspace/R-latplan/r_latplan_datasets/hanoi/hanoi_complete_clean_faultless_withoutTI_withoutOne/dataMinusOne.p"

# load dataset for the specific experiment
loaded_data = load_dataset(path_to_dataset)
train_set_no_dupp = loaded_data["train_set_no_dupp_processed"]

# GROUP THE TRANSITIONS BY THEIR HIGH LVL ACTION
dico_transitions_per_high_lvl_actions = {}

# loop over the train set (without dupplicate)
# AND group the transitions into their respective High Level Actions
# [all_pairs_of_images_processed_gaussian20[i], all_actions_one_hot[i], all_high_lvl_actions_one_hot[i]]
for ii, ele in enumerate(train_set_no_dupp):

    # if the looped transition high level action is not a key of dico_transitions_per_high_lvl_actions
    if np.argmax(ele[2]) not in dico_transitions_per_high_lvl_actions:

        # add the key (the high lvl action index)
        dico_transitions_per_high_lvl_actions[np.argmax(ele[2])] = {}

    # if the hash is not in the keys of the dico of the high lvl action
    # i.e. if the transition was not added as a transition for this high lvl action
    if np.argmax(ele[1]) not in dico_transitions_per_high_lvl_actions[np.argmax(ele[2])]:
        
        dico_transitions_per_high_lvl_actions[np.argmax(ele[2])][np.argmax(ele[1])] = {
            "preprocessed" : ele[0],
            #"preprocessed": "aaa"
            # "reduced" : train_set_no_dupp_orig[ii][0],
            # "onehot": ele[2],
            #"max_diff_in_bits_": ele[2]
        }


# with open("highLvlLowlvl.json", "w") as json_file:
#     json.dump(dico_transitions_per_high_lvl_actions, json_file, indent=4) 

# exit()


### 2) construct the "X" and the "Y" for each high lvl action (i.e. the preconditions (pos/del) of each transitions)

# load pos and neg CSV ('aligned' coz i) we removed unecessary duplicates in the effects AND
# ii) we tagged the actions without effects with 9s (preconds were also tagged with 9s)
pos_preconds = np.loadtxt("pos_preconds_aligned.csv", delimiter=' ', dtype=int)
neg_preconds = np.loadtxt("neg_preconds_aligned.csv", delimiter=' ', dtype=int)

add_effs = np.loadtxt("add_effs_aligned.csv", delimiter=' ', dtype=int)
del_effs = np.loadtxt("del_effs_aligned.csv", delimiter=' ', dtype=int)

effects_set = []
for i in range(16):
    effects_set.append(f"add_{i}")
for i in range(16):
    effects_set.append(f"del_{i}")
preconds_perEff_perAction = {}


# feature_names_part_1 = [f"(z{i})" for i in range(16)]
# feature_names_part_2 = [f"not (z{i})" for i in range(16)]
# feature_names = feature_names_part_1
# feature_names.extend(feature_names_part_2)


# z_0_1  z_0_0  z_1_1  z_1_0  z_2_1  z_2_0
feature_names = []

for i in range(16):
    feature_names.append(f"(z_{i}_1)")
    feature_names.append(f"(z_{i}_0)")

# print(len(feature_names))

# print(feature_names)

# exit()







def format_precond(precond):
    splits = precond.split('_')
    if int(precond[-1]) == 0:
        return '(not ('+str(splits[0])+str(splits[1])+'))'
    elif int(precond[-1]) == 1:
        return '('+str(splits[0])+str(splits[1])+')'


# loop over high level actions
for num_action in range(0, 22):

    preconds_perEff_perAction[num_action] = {}

    ############### Constructing the X ###############

    action_transitions_preconds = []

    # doing the union of all preconditions for this action
    # (actually, for each transition representing this action)
    for trans_id in dico_transitions_per_high_lvl_actions[num_action].keys():
        preconds_union = []
        preconds_union.extend(pos_preconds[trans_id])
        preconds_union.extend(neg_preconds[trans_id])
        action_transitions_preconds.append(preconds_union)

    # action_transitions_preconds of shape (48, 100)
    # i.e. pour chque transition, le vecteur concaténé des preconds pos et neg
    X = np.array(action_transitions_preconds)


 

    X = pd.DataFrame(X)

    # action_transitions_preconds of shape (202, 100) , i.e. 202 is the number of transitions and 100 the number of possible 
    # preconditions (first 50th are for the positive ones, 0 must be interpreted as "we don't care", 1 means the precond is here)

    # put a Label on rows and columns
    new_columns = []
    for i in range(16):
        new_columns.append(f"z_{i}")
    for i in range(16):
        new_columns.append(f"not(z_{i})")

    X.columns = new_columns[:len(X.columns)] # Use slicing to handle cases where X might have fewer columns

    X.index = list(dico_transitions_per_high_lvl_actions[num_action].keys())



    #X = X.drop(index=[10, 19])
    indices_with_nine = X[X.isin([9]).any(axis=1)].index

    X = X.drop(index=indices_with_nine)
    #Y = Y.drop(index=indices_with_nine)



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

    #new_X = pd.DataFrame(X)


    #print(df)

    # df_oh = pd.get_dummies(
    #     data=new_X)

    # print(df_oh)
    # enc = OneHotEncoder(handle_unknown='ignore')
    # enc.fit(X)
    # print(len(enc.categories_))
    #exit()

    ############### Constructing the Y ###############

    # for each transition, the effects

    action_transitions_effects = []
    # doing the union of all EFFECTS for this action
    # (actually, for each transition representing this action)
    for trans_id in dico_transitions_per_high_lvl_actions[num_action].keys():
        effects_union = []
        effects_union.extend(add_effs[trans_id])
        effects_union.extend(del_effs[trans_id])
        action_transitions_effects.append(effects_union)

    Y = np.array(action_transitions_effects)
    Y = pd.DataFrame(Y)

    # put a Label on rows and columns
    new_columns = []
    for i in range(16):
        new_columns.append(f"add_{i}")
    for i in range(16):
        new_columns.append(f"del_{i}")

    Y.columns = new_columns[:len(Y.columns)] # Use slicing to handle cases where X might have fewer columns
        
    Y.index = list(dico_transitions_per_high_lvl_actions[num_action].keys())

    Y = Y.drop(index=indices_with_nine)

    print(type(new_X.iloc[0][0]))

    new_X.replace(1, "1", inplace=True)
    new_X.replace(0, "0", inplace=True)
    #new_X.replace("?", "C", inplace=True)

    onehot_encoder = OneHotEncoder(sparse=False, categories = [["1", "0", "?"] for i in range(16)])  # 'first' to drop the first category to avoid multicollinearity

    onehot_encoded = onehot_encoder.fit_transform(new_X) # Fit and transform the data

    print(onehot_encoded.shape)

    new_X = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(['z_'+str(i) for i in range(16)]))



    # removing the "?" columns
    new_X = new_X.loc[:, ~new_X.columns.str.contains(r'\?')]



    


    ### 3) train a sklearn MultiOutputClassifier
    X_train, X_test, Y_train, Y_test = train_test_split(new_X, Y, test_size=0.2, random_state=42)



    # model = OneVsRestClassifier(CatBoostClassifier(iterations=20, depth=12, learning_rate=0.1, verbose=True, cat_features=cat_features))
    # model.fit(X_train, Y_train)
    print(new_X)

    # y_pred = model.predict(X_test)

    # # Calculate accuracy and F1 score
    # accuracy = accuracy_score(Y_test, y_pred)
    # f1 = f1_score(Y_test, y_pred, average='macro')

    # print(f'Accuracy: {accuracy}')
    # print(f'F1 Score: {f1}')
    # exit()


    # train_dataset = Pool(data=X_train,
    #                     label=Y_train,
    #                     cat_features=cat_features)

    # cb = CatBoost({'iterations': 10})
    # cb.fit(train_dataset)

    #Fit model
    #model.fit(X_train, Y_train, cat_features)

    # ############# SKLEARN MULTI OUTPUT
    # print("X Y and")
    # print(X_train.shape)
    # print(Y_train.shape)
    # exit()

    clf = MultiOutputClassifier(DecisionTreeClassifier(criterion='entropy', random_state=42))
    clf.fit(X_train, Y_train)

    # Tester le modèle
    Y_pred = clf.predict(X_test)

    # Évaluer la performance du modèle (par exemple, par une accuracy moyenne)
    accuracy = np.mean(Y_pred == Y_test)
    print(f"Accuracy: {accuracy}")



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

                path += ["class is : "+str(np.argmax(tree_.value[node]))]
                paths += [path]
                
        recurse(0, path, paths)
        return paths



    # Pour chaque effet, extraire et afficher les règles d'arbres de décision
    for idx, estimator in enumerate(clf.estimators_):
        #print(f"Rules for Effect {idx + 1}:")

        print("effects_seteffects_seteffects_set")

        effect_name = effects_set[idx]
        preconds_perEff_perAction[num_action][effect_name] = []
        # print(f"Rules for Effect {effect_name}:")
        
        #tree_rules = export_text(estimator, feature_names=feature_names)


        # plt.figure(figsize=(12, 12))
        # plot_tree(estimator, 
        #         feature_names=feature_names, 
        #         label= 'all',
        #         #class_names = ["0", "1"],
        #         class_names=[f'Class {k}' for k in range(2)],  # Assuming binary classification for each output
        #         filled=False, 
        #         rounded=True,
        #         impurity = False
        #         )
        # plt.title(f"Decision Tree for Effect {effect_name}", fontsize = 40)



        # file_name = f"DTs_action_{str(num_action)}/decision_tree_effect_{effect_name}.png"
        # plt.savefig(file_name, format="png", dpi=300)  



        clf__ = estimator


        # Ugly paths are ALL the path of the current tree (one tree per effect)
        ugly_paths = get_rules(clf__, feature_names, None)


        #print("ugly_paths is {}".format(str(ugly_paths)))
        # [['((z_5_0) <= 0.5)', 'class is : 0'], ['((z_5_0) > 0.5)', '((z_4_1) <= 0.5)', 'class is : 0'], ['((z_5_0) > 0.5)', '((z_4_1) > 0.5)', 'class is : 1']]s



        beauty_paths = []
        for pathh in ugly_paths:

            if pathh[-1] == 'class is : 0':
                continue
            elif pathh[-1] == 'class is : 1':

                tmp = []
                for cond in pathh[:-1]:
                    if ">" in cond:
                        #tmp.append(cond.split(" > ")[0][1:])
                        tmp.append(format_precond(cond.split(" > ")[0].replace("(", "").replace(")", "")))
                        # print("loool")
                        # print(cond)
                        # print(format_precond(cond.split(" > ")[0].replace("(", "").replace(")", "")))

                    else:
                        tmp = []
                        break
                    
                if len(tmp) > 0:
                    beauty_paths.append(tmp)

        # print("beauty_paths for effect {}".format(str(effect_name)))
        # print(beauty_paths)
        # continue


        

        print("=" * 50)  # Separator between trees

        # beaut_preconds = []
        # for preconds_set in beauty_paths:
        #     if len(preconds_set) > 0:

        #         tmp_preconds = []
        #         for pr in preconds_set[:-1]:
        #             #print("was here")
        #             index_of_precond = pr.split("_")[1]
        #             tmp_preconds.append(preconds_names[int(index_of_precond)])
        #         beaut_preconds.append(tmp_preconds)

        preconds_perEff_perAction[num_action][effect_name] = beauty_paths

### 4) generate the conditional pddl


from collections import defaultdict

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



#####################
# writting the PDDL 
#######################

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


    ### 1) vois combien d'effect par action

    ### 2) verifie que ce nbre d'effects

    for num_action in range(0, 22):

        reformatted_dict, updated_D = reformat_dict(preconds_perEff_perAction[num_action])
        merged_dict = merge_and_cleanup(reformatted_dict, updated_D)

        #print("merged_dictmerged_dictmerged_dictmerged_dictmerged_dict")

        f.write("(:action a"+str(num_action)+"\n")
        f.write("   :parameters ()\n")
        f.write("   :precondition ()\n")
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
                        str_before = two_tabs_space + "(and \n"
                        str_after = two_tabs_space + ")\n"
                    #tmp_str += str_before
                    tmp_group_of_effects = ''
                    for eff_name in  list(eff_group):
                        

                        if eff_name.split('_')[0] == 'del':
                            eff_name = "(not (z"+eff_name.split('_')[1]+"))"
                        else:
                            eff_name = "(z"+eff_name.split('_')[1]+")"
                        
                        tmp_group_of_effects += two_tabs_space+eff_name + "\n"

                    tmp_str += str_before + tmp_group_of_effects + str_after
                    tmp_str += two_tabs_space+")\n"

                    str_preconds += tmp_str


            f.write(str_preconds)



        f.write("   )\n")
        f.write(")\n")
        #[effect_name] = beaut_preconds
        # 
    
    f.write(")\n")

