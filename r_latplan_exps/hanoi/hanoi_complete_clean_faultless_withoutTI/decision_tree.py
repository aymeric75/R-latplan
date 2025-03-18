
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz, plot_tree
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
### retrieve the transitions ids for each action
import os 



import pickle



def save_dataset(dire, X, Y):
    data = {
        "X": X,
        "Y": Y
    }
    if not os.path.exists(dire):
        os.makedirs(dire) 
    filename = "data_set.p"
    with open(dire+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)




dic_action_transitions = {} 



dic_shap_perEffect_perTrans_perAction = {}

for num_action in range(0, 22):

    dic_action_transitions["action_"+str(num_action)] = []

    dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)] = {}

    #with open("shap_vals_persisting_effects_removed/action_"+str(num_action)+".txt", "r") as file:
    with open("shap_vals_persisting_effects_removed/action_"+str(num_action)+"_withEmptySet.txt", "r") as file:

        for line in file:

            if "transition" in line:
                last_key = int(line.split(" ")[1].strip())
                dic_action_transitions["action_"+str(num_action)].append(last_key)
                dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)][last_key] = {}


            elif "add_" in line or "del_" in line:
                if len(dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)][last_key].values()) == 0:
                    dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)][last_key] = {}
                dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)][last_key][line.split(" ")[0].strip()] = float(line.split(" ")[1].strip())



# print("dic_action_transitionsdic_action_transitions dic_action_transitionsdic_action_transitions")
# print(dic_action_transitions['action_0'])
# exit()


###  for each transition ID, retrieve (from the *4 files) the pos/neg preconditions



pos_preconds = np.loadtxt("action_pos4.csv", delimiter=' ', dtype=int)
neg_preconds = np.loadtxt("action_neg4.csv", delimiter=' ', dtype=int)

# dic_transition_pos_preconds = {}
# dic_transition_neg_preconds = {}


# for trans_id in len(dic_transition_pos_preconds):

#     dic_transition_pos_preconds[trans_id] = pos_preconds[]


preconds_names = ["(z"+str(i)+")" for i in range(50)]
neg_preconds_names = ["(not (z"+str(i)+"))" for i in range(50)]

preconds_names.extend(neg_preconds_names)

print(preconds_names)

preconds_perEff_perAction = {}

for num_action in range(0, 22):

    preconds_perEff_perAction[num_action] = {}


    ########################################################
    #### Building the X   ##################################
    #### now considering ONE action (0)
    ########################################################

    action_transitions_preconds = []



    for trans_id in dic_action_transitions["action_"+str(num_action)]:
        preconds_union = []
        preconds_union.extend(pos_preconds[trans_id])
        preconds_union.extend(neg_preconds[trans_id])
        action_transitions_preconds.append(preconds_union)

    # action_transitions_preconds of shape (48, 100)
    # i.e. pour chque transition, le vecteur concaténé des preconds pos et neg
    action_transitions_preconds = np.array(action_transitions_preconds)


    print(action_transitions_preconds.shape)




    #### Building the Y

    # retrieve all the unique effects names, put them as key in a dict

    # total set of effects for the high lvl action (here action 0)
    effects_set = []
    for trans_id, dico_effects in dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)].items():

        for eff_name, shap_val in dico_effects.items():
            if (eff_name not in effects_set): # and shap_val > 0 :
                effects_set.append(eff_name)


    # set of the effects, sorted alphabetically and by index (e.g. add_0 then add_1 etc)
    effects_set = sorted(effects_set, key=lambda x: (x.split('_')[0], int(x.split('_')[1])))

    print("effects_set siz e")
    print(len(effects_set))

    ##  pos ( 1 0 ... )  (50)
    ##  neg ( 0 0 ... )  (50)

    ####   

    # Rules for Effect 1:
    # |--- precondition_56 <= 0.50         
    # |   |--- precondition_24 <= 0.50      !!!! DONT CONSIDER THIS 
    # |   |   |--- class: 1
    # |   |--- precondition_24 >  0.50
    # |   |   |--- class: 0
    # |--- precondition_56 >  0.50      !!!!! THIS YES 
    # |   |--- class: 0




    # Making the table where ROWS = transition preconditions,   COL = SHAP of effects
    list_transIdKey_shapEffectsValues = [] # (#transitions, #effects)
    for j, (trans_id, dico_effects) in enumerate(dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)].items()):

        #list_transIdKey_shapEffectsValues[j] = np.zeros(len(effects_set))

        list_transIdKey_shapEffectsValues.append([0 for ii in range(len(effects_set))])

        for i, effName in enumerate(effects_set):

            
            if effName in dico_effects.keys():
                if dico_effects[effName] > 0:
                    list_transIdKey_shapEffectsValues[j][i] = 1


    print("list_transIdKey_shapEffectsValueslist_transIdKey_shapEffectsValues")
    list_transIdKey_shapEffectsValues = np.array(list_transIdKey_shapEffectsValues)


    print(list_transIdKey_shapEffectsValues.shape)

    # (48, 56), for each transition, a binary vector that says if this or that
    # effect as a "true" shap value (i.e. if it should be considered)
    print(list_transIdKey_shapEffectsValues.shape)


    X = action_transitions_preconds

    print("X   ")
    print(X.shape) # (48, 100)

    Y = list_transIdKey_shapEffectsValues
    print("Y shape") # (48, 56)
    print(Y.shape)
    print(effects_set)


    np.savetxt("X.txt", X, delimiter=" ", fmt="%d")

    np.savetxt("Y.txt", Y, delimiter=" ", fmt="%d")


    save_dataset("./", X, Y)

    print(X.shape)
    print(Y.shape)

    exit()


    # Diviser les données en ensembles d’entraînement et de test pour valider le modèle
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = MultiOutputClassifier(DecisionTreeClassifier(criterion='entropy', random_state=42))
    clf.fit(X_train, Y_train)

    # Tester le modèle
    Y_pred = clf.predict(X_test)

    # Évaluer la performance du modèle (par exemple, par une accuracy moyenne)
    accuracy = np.mean(Y_pred == Y_test)
    print(f"Accuracy: {accuracy}")
    print()
    #exit()
    from sklearn.tree import _tree


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

    # Extract and display paths
    #feature_names = [f"precondition_{i}" for i in range(X.shape[1])]

    feature_names_part_1 = [f"(z{i})" for i in range(X.shape[1]//2)]
    feature_names_part_2 = [f"not (z{i})" for i in range(X.shape[1]//2)]

    feature_names = feature_names_part_1
    feature_names.extend(feature_names_part_2)

    # print(feature_names)
    # exit()

    import re

    #export_text(estimator, feature_names=[f"precondition_{i}" for i in range(X.shape[1])])




    # # PDF file to save all the decision trees
    # output_pdf_file = "decision_trees.pdf"

    # with PdfPages(output_pdf_file) as pdf:
    #     for i, (estimator, effect_name) in enumerate(zip(clf.estimators_, effects_set)):
    #         plt.figure(figsize=(12, 12))
    #         plot_tree(estimator, 
    #                 feature_names=feature_names, 
    #                 label='all',
    #                 class_names=[f'Class {k}' for k in range(2)],  # Assuming binary classification for each output
    #                 filled=False, rounded=True,
    #                 impurity=False)
    #         plt.title(f"Decision Tree for Effect {effect_name}", fontsize=40)
            
    #         # Save the current figure to the PDF
    #         pdf.savefig()  # This saves the current figure into the PDF
    #         plt.close()  # Close the figure to free up memory

    # print(f"All decision trees saved into {output_pdf_file}")


    # exit()


    import math


    # PDF file to save all the decision trees
    output_pdf_file = "decision_trees_GRAPH.pdf"

    # Determine grid size (square layout)
    num_trees = len(clf.estimators_)
    grid_size = math.ceil(math.sqrt(num_trees))  # Closest square layout

    # Create a figure large enough to hold all subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()  # Flatten to iterate easily

    print("lllllaaaa")

    for i, (estimator, effect_name) in enumerate(zip(clf.estimators_, effects_set)):
        ax = axes[i]
        plot_tree(estimator, 
                feature_names=feature_names, 
                label='all',
                class_names=[f'Class {k}' for k in range(2)],  # Assuming binary classification for each output
                filled=False, rounded=True,
                impurity=False,
                ax=ax)
        ax.set_title(f"Effect: {effect_name}", fontsize=8)

        # tree_rules = export_text(estimator, feature_names=feature_names)

        # ax.text(0.5, 0.5, tree_rules, fontsize=6, ha='left', va='top', wrap=True, 
        # fontstretch=500)

        # ax.set_title(f"Effect: {effect_name}", fontsize=8)
        # ax.axis('off')


    # Hide any extra subplots (if there are fewer trees than grid spaces)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save to PDF
    fig.savefig(output_pdf_file, format="pdf", dpi=300)
    plt.close(fig)

    print(f"All decision trees saved into {output_pdf_file} in a square layout")


    exit()




    # Pour chaque effet, extraire et afficher les règles d'arbres de décision
    for idx, estimator in enumerate(clf.estimators_):
        #print(f"Rules for Effect {idx + 1}:")

        effect_name = effects_set[idx]
        preconds_perEff_perAction[num_action][effect_name] = []
        print(f"Rules for Effect {effect_name}:")



        tree_rules = export_text(estimator, feature_names=feature_names)
 

        plt.figure(figsize=(12, 12))
        plot_tree(estimator, 
                feature_names=feature_names, 
                label= 'all',
                #class_names = ["0", "1"],
                class_names=[f'Class {k}' for k in range(2)],  # Assuming binary classification for each output
                filled=False, rounded=True,
                impurity = False
                )
        plt.title(f"Decision Tree for Effect {effect_name}", fontsize = 40)

        file_name = f"decision_tree_effect_{effect_name}.png"
        plt.savefig(file_name, format="png", dpi=300)  

        exit()
        # dot_data = export_graphviz(estimator, 
        #                         out_file=None, 
        #                         feature_names=feature_names, 
        #                         #class_names=[f'Class {k}' for k in range(Y.shape[1])],
        #                         filled=True, rounded=True, special_characters=True)
        # # Visualize using graphviz
        # graph = graphviz.Source(dot_data)
        # graph.render(f"tree_output_{i}")  # Save each tree visualization as a file

        # exit()

        continue

        clf = estimator
        ugly_paths = get_rules(clf, feature_names, None)



        beauty_paths = []
        for pathh in ugly_paths:
            tmp_path = []
            has_inf_or_equal = False
            for node in pathh:
                if "<= 0.5" in node:
                    has_inf_or_equal = True
            if not has_inf_or_equal:
                for node in pathh:
                    if "> 0.5" in node:
                        to_add = re.findall(r"precondition_\d+", node)
                        if len(to_add) > 0:
                            tmp_path.append(to_add[0])
                    elif "class is" in node:
                        tmp_path.append(node)
            if len(tmp_path) > 0 and "class is : 1" == tmp_path[-1] :
                beauty_paths.append(tmp_path)

        print()
        print(beauty_paths)
        print("=" * 50)  # Separator between trees
        print()
        #print(beauty_paths)
        #exit()

        beaut_preconds = []
        for preconds_set in beauty_paths:
            if len(preconds_set) > 0:

                tmp_preconds = []
                for pr in preconds_set[:-1]:
                    #print("was here")
                    index_of_precond = pr.split("_")[1]
                    tmp_preconds.append(preconds_names[int(index_of_precond)])
                beaut_preconds.append(tmp_preconds)

        preconds_perEff_perAction[num_action][effect_name] = beaut_preconds



    #print(preconds_perEff_perAction)
    #exit()



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

# # Example usage:
# D = {
#     'a': [[1, 2], [3, 4]],
#     'b': [[3, 4], [5, 6]],
#     'c': [[7, 8]],
#     'd': [[1, 2], [7, 8]],
#     'e': [[9, 0]],
# }



# print("Merged and Cleaned Dictionary:")
# print(merged_dict)




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
    
    f.write(")\n")