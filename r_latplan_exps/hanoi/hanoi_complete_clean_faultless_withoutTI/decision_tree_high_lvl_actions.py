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



##### 1) load the conditional PDDL



domainfile="domain_ORIGINAL.pddl"
problemfile="problem.pddl"
translate_path = os.path.join("/workspace/R-latplan", "downward", "src", "translate")
sys.path.insert(0, translate_path)

try:
    import FDgrounder
    from FDgrounder import pddl_parser as pddl_pars
    task = pddl_pars.open(
        domain_filename=domainfile, task_filename=problemfile) # options.task


    list_of_lists = []
    for action_id, act in enumerate(task.actions):

        


    #     if str(trans_id) in all_lowLvl_ids:

    #         tmp_act_precond_parts = list(act.precondition.parts)
    #         tmp_list = []
                
    #         for precond in tmp_act_precond_parts:
    #             transformed_name_ = friendly_effect_name(precond)

    #             tmp_list.append(transformed_name_)

    #         list_of_lists.append(tmp_list)

    # dico[num] = list_of_lists

finally:
    # Restore sys.path to avoid conflicts
    sys.path.pop(0)






##### 2) for each high lvl action, retrieve the set of conditions


#### 3)      for each atom chek if the atom belongs to one of the sets


#####      4)     IF YES, remove it from the X


#####           5) train the DT of the X wrt high lvl action




def load_dataset(path_to_file):
    import pickle
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    # print("path_to_file path_to_file")
    # print(path_to_file)
    # exit()
    return loaded_data




def friendly_effect_name(literal):       
    integer = int(''.join(x for x in str(literal) if x.isdigit()))
    transformed_name = ""
    if "Negated" in str(literal):
        transformed_name += "del_"+str(integer)
    else:
        transformed_name += "add_"+str(integer)
    return transformed_name


path_to_dataset = "/workspace/R-latplan/r_latplan_datasets/hanoi/hanoi_complete_clean_faultless_withoutTI/data.p"

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




# train a DT per high lvl action that says, in function of the preconds if the action should apply or not
#
#
#   for one DT (one action)
#
#           the     X is all the transitions preconditions
#
#
#              the Y is a (#transitions) shaped vector where 1 when  it correspond to the current action, 0 otherwise

#                   in the end, the DT says, in function of some preconds, if applky yher action or not

#



# le "X" 
all_transitions_preconds = np.zeros((len(pos_preconds), len(pos_preconds[0])*2))

all_Ys = np.zeros((len(pos_preconds), 22))


for num_action in range(0, 22):

    for trans_id in dico_transitions_per_high_lvl_actions[num_action].keys():

        all_Ys[trans_id][num_action] = 1

        #print(all_transitions_preconds[trans_id])
        preconds_union = []
        preconds_union.extend(pos_preconds[trans_id])
        preconds_union.extend(neg_preconds[trans_id])
        all_transitions_preconds[trans_id] += preconds_union


all_transitions_preconds = pd.DataFrame(all_transitions_preconds)


new_columns = []
for i in range(16):
    new_columns.append(f"z_{i}")
for i in range(16):
    new_columns.append(f"not(z_{i})")

all_transitions_preconds.columns = new_columns[:len(all_transitions_preconds.columns)] # Use slicing to handle cases where X might have fewer columns
print(all_transitions_preconds.head())


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

print(new_entire_X.head())

print()

new_entire_X.replace(1, "1", inplace=True)
new_entire_X.replace(0, "0", inplace=True)

onehot_encoder = OneHotEncoder(sparse=False, categories = [["1", "0", "?"] for i in range(16)])  # 'first' to drop the first category to avoid multicollinearity

onehot_encoded = onehot_encoder.fit_transform(new_entire_X) # Fit and transform the data

new_entire_X = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(['z_'+str(i) for i in range(16)]))



# removing the "?" columns
new_entire_X = new_entire_X.loc[:, ~new_entire_X.columns.str.contains(r'\?')]


#all_transitions_preconds = pd.DataFrame(all_transitions_preconds)

# print("all_transitions_preconds head")
# print(all_transitions_preconds[:6])


# exit()
# le "Y", pour chaque 


# np.zeros()
# dico_transitions_per_high_lvl_actions[num_action].keys()





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
            # print("aaagggg")
            # print(node)
            # print(tree_.value[node])
            # print(tree_.n_node_samples[node])
            # print(tree_.impurity[node])
            path += ["class is : "+str(np.argmax(tree_.value[node]))]
            paths += [path]
            
    recurse(0, path, paths)
    return paths

# loop over high level actions
for num_action in range(0, 22):


    # For each action, do the DT of preconds for knowning if apply the action or not

    clf2 = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf2.fit(new_entire_X, all_Ys[:, num_action])

    ugly_paths = get_rules(clf2, feature_names, None)
    print("finished")
    print(ugly_paths)
    print(len(ugly_paths))

    # 
    #   
    #
    #



    plt.figure(figsize=(72, 72))
    plot_tree(clf2, 
            feature_names=feature_names, 
            label= 'all',
            class_names=[f'Class {k}' for k in range(2)], 
            filled=True, 
            rounded=True,
            impurity = True
            )

    

    say_one_positive_node = ""
    
    # Ugly paths are ALL the path of the current tree (one tree per effect)
    ugly_paths = get_rules(clf2, feature_names, None)
    # if len(ugly_paths) == 1 and len(ugly_paths[0]) == 1:
    #     one_node = True
    #     summ = int(Y_train[effect_name].sum())
    #     if summ > 0:
    #         say_one_positive_node = "CLASS 1"
    #     else:
    #         say_one_positive_node = "CLASS 0"

    plt.title(f"Decision Tree for Action {num_action}, {say_one_positive_node}", fontsize = 40)
    #file_name = f"DTs_action_{str(num_action)}/decision_tree_HIGH_LEVEL_{num_action}.png"
    file_name = f"decision_tree_HIGH_LEVEL_{num_action}.png"
    plt.savefig(file_name, format="png", dpi=300)  

    exit()
    continue




    counterr = 0
    for path in ugly_paths:

        if "class is : 1" in path[-1]:
            counterr += 1
    
    print("counterr is {}".format(str(counterr)))
    exit()

    # one_node = False 
    # the_one_node_is_true = False

    # add_stuff = False

    # # if the "tree" is just one node, test if it is a "class 1" node
    # if len(ugly_paths) == 1 and len(ugly_paths[0]) == 1:
    #     one_node = True
    #     summ = int(Y_train[effect_name].sum())

    #     if summ > 0:
    #         ugly_paths[0][0].replace('0', '1')
    #         the_one_node_is_true = True


    # beauty_paths = []

    # beauty_paths_class_1 = 0

    # if not one_node:
    #     add_stuff = True
    #     for pathh in ugly_paths:

    #         if pathh[-1] == 'class is : 0':
    #             continue
    #         elif pathh[-1] == 'class is : 1': # and len(pathh[:-1]) > 2:
                
    #             tmp = []
    #             for cond in pathh[:-1]:
    #                 if ">" in cond:
    #                     tmp.append(format_precond(cond.split(" > ")[0].replace("(", "").replace(")", "")))
    #                 else: # "<=" in cond
    #                     tmp.append(format_precond(cond.split(" <= ")[0].replace("(", "").replace(")", ""), reverse=True))
    #                     #tmp.append(format_precond(cond.split(" <= ")[0].replace("(", "").replace(")", "").replace("not", "")))
    #                     #break
                    
    #             if len(tmp) > 0:
    #                 beauty_paths.append(tmp)

    #     #preconds_perEff_perAction[num_action][effect_name] = beauty_paths

    # elif the_one_node_is_true:
    #     tmp = []
    #     beauty_paths.append(tmp)
    #     add_stuff = True
    #     #preconds_perEff_perAction[num_action][effect_name] = []

    # print("beauty pathss")


    # print(beauty_paths)

    # print("=" * 50)  # Separator between trees


    # exit()


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

    
    print(X.head())


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

    ### 3) train a sklearn MultiOutputClassifier                # test_size=0.2
    X_train, X_test, Y_train, Y_test = train_test_split(new_X, Y, test_size=0.2, random_state=42)

    # X_train = new_X
    # Y_train = Y

    print(Y_train.shape) # (193, 32)
    print(Y_train.head())

    # INSTEAD JE VEUX, poru chaque transition, une sorte de one hot horizonrtal qui

    # model = OneVsRestClassifier(CatBoostClassifier(iterations=20, depth=12, learning_rate=0.1, verbose=True, cat_features=cat_features))
    # model.fit(X_train, Y_train)
    print(new_X)


    # ############# SKLEARN MULTI OUTPUT
    # print("X Y and")
    # print(X_train.shape)

 
    clf = MultiOutputClassifier(DecisionTreeClassifier(criterion='entropy', random_state=42))
    clf.fit(X_train, Y_train)

    # Tester le modèle
    Y_pred = clf.predict(X_test)

    # Évaluer la performance du modèle (par exemple, par une accuracy moyenne)
    accuracy = np.mean(Y_pred == Y_test)
    #print(f"Accuracy: {accuracy}")


    # TRAINING THE DT FOR preconds against action 1 or 0
    # first make the data pandas dataframes sim to for the other tree
    



    # Pour chaque effet, extraire et afficher les règles d'arbres de décision
    for idx, estimator in enumerate(clf.estimators_):
        #print(f"Rules for Effect {idx + 1}:")

        #print("effects_seteffects_seteffects_set")

        effect_name = effects_set[idx]
        #preconds_perEff_perAction[num_action][effect_name] = []
        # print(f"Rules for Effect {effect_name}:")
        
        #tree_rules = export_text(estimator, feature_names=feature_names)

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
        #     else:
        #         say_one_positive_node = "CLASS 0"

        # plt.title(f"Decision Tree for Effect {effect_name}, {say_one_positive_node}", fontsize = 40)
        # file_name = f"DTs_action_{str(num_action)}/decision_tree_effect_{effect_name}.png"
        # plt.savefig(file_name, format="png", dpi=300)  

        # continue


        clf__ = estimator


        # Ugly paths are ALL the path of the current tree (one tree per effect)
        ugly_paths = get_rules(clf__, feature_names, None)

        #print(ugly_paths)

        # [['((z_5_0) <= 0.5)', 'class is : 0'], ['((z_5_0) > 0.5)', '((z_4_1) <= 0.5)', 'class is : 0'], ['((z_5_0) > 0.5)', '((z_4_1) > 0.5)', 'class is : 1']]s

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

                if pathh[-1] == 'class is : 0':
                    continue
                elif pathh[-1] == 'class is : 1': # and len(pathh[:-1]) > 2:

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
        print("=" * 50)  # Separator between trees




        # if num_action == 7 and effect_name == "add_10":
        #     print("NUM ACTION IS 7")
        #     print("EFFECT IS add_10")
        #     print(beauty_paths)
        #     exit()
        # if add_stuff and len(beauty_paths) > 0:
        #     preconds_perEff_perAction[num_action][effect_name] = beauty_paths


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







from collections import defaultdict

# def factorize_dict(input_dict):
#     list_to_keys = defaultdict(set)
#     new_dict = defaultdict(list)
    
#     # Reverse mapping from lists to keys
#     for key, lists in input_dict.items():
#         if not lists:
#             new_dict[key] = []  # Preserve empty lists
#         for lst in lists:
#             tuple_lst = tuple(sorted(lst))  # Convert list to tuple for immutability and sorting for consistency
#             list_to_keys[tuple_lst].add(key)
    
#     # Construct the new factorized dictionary
#     processed_keys = set()
#     for lst, keys in list_to_keys.items():
#         keys_tuple = tuple(sorted(keys))
#         new_dict[keys_tuple].append(list(lst))
#         processed_keys.update(keys)
    
#     # Convert keys tuples to appropriate format
#     factorized_dict = {}
#     for keys, lists in new_dict.items():
#         if isinstance(keys, str):  # Keep individual keys with empty lists
#             factorized_dict[keys] = lists
#         elif len(keys) == 1:
#             factorized_dict[keys[0]] = lists
#         else:
#             factorized_dict["_".join(keys)] = lists
    
#     return factorized_dict




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
    for lst, keys in list_to_keys.items():
        keys_tuple = tuple(sorted(keys))
        new_dict[keys_tuple].append(list(lst))
        processed_keys.update(keys)
    
    # Convert keys tuples to appropriate format
    factorized_dict = {}
    for keys, lists in new_dict.items():
        factorized_dict[tuple(keys)] = lists  # Ensure all keys are stored as tuples
    
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

    #dico_disjunctions_per_high_lvl_action_ = dico_disjunctions_per_high_lvl_action()
    dico_disjunctions_simple_per_high_lvl_action_ = dico_disjunctions_simple_per_high_lvl_action()

    for num_action in range(0, 22):

        

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


        print("DEBUT 1 ")
        # #reformatted_dict, updated_D = reformat_dict(preconds_perEff_perAction[num_action])


        merged_dict = factorize_dict(preconds_perEff_perAction[num_action])
        #print(merged_dict)


        f.write("(:action a"+str(num_action)+"\n")
        f.write("   :parameters ()\n")
        #f.write("   :precondition ()\n")



        

        f.write("   :precondition (OR ")

        # # CASE : precond is an OR of all low lvl preconds sets
        
        # for liste in dico_disjunctions_per_high_lvl_action_[num_action]:
        #     tmp_str = "(AND"
        #     for ele in liste:
        #         ele = transfo_precond_effect_name_to_pddl(ele)
        #         tmp_str += " "+ele 
        #     tmp_str += ")"
        #     f.write(" "+tmp_str)
        
        # CASE : precond is an OR of the set of all lvl preconds 
        #dico_disjunctions_simple_per_high_lvl_action
        for liste in dico_disjunctions_simple_per_high_lvl_action_[num_action]:

            tmp_str = ""
            for ele in liste:
                ele = transfo_precond_effect_name_to_pddl(ele)
                tmp_str += " "+ele 
            tmp_str += ""
            f.write(" "+tmp_str)

        f.write(")\n") # closing the (



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

