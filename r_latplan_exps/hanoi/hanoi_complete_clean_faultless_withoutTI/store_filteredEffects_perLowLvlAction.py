
import numpy as np

def load_dataset(path_to_file):
    import pickle
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    # print("path_to_file path_to_file")
    # print(path_to_file)
    # exit()
    return loaded_data



# for each high lvl action retrieve its set of filtered effects 
# (goes through action_*_main_effects_withEmptySet_corrected.txt etc)

dico_highAction_filteredEffects = {}




for num_action in range(0, 22):

    dico_highAction_filteredEffects[num_action] = []

    with open("shap_vals_persisting_effects_removed/action_"+str(num_action)+"_main_effects_withEmptySet_corrected.txt", "r") as file:

        for line in file:
            dico_highAction_filteredEffects[num_action].append(line.strip())
        


# dico high lvl / low levels IDs



# 
dataset_fold = None


dataset_fold = "/workspace/R-latplan/r_latplan_datasets"

# DATASETS FOLDERS
dataset_aux_json_folder_base = dataset_fold+"/"+"hanoi"
dataset_aux_json_folder_exp = dataset_fold+"/"+"hanoi" + "/" + "hanoi_complete_clean_faultless_withoutTI"

# EXPERIMENTS FOLDERS
exp_aux_json_folder = None

exp_aux_json_folder = "r_latplan_exps/" +"hanoi" + "/" + "hanoi_complete_clean_faultless_withoutTI"

# 
path_to_dataset = dataset_aux_json_folder_exp +"/data.p"


# load dataset for the specific experiment
loaded_data = load_dataset(path_to_dataset)

train_set = loaded_data["train_set"] 
test_val_set = loaded_data["test_val_set"] 
all_pairs_of_images_reduced_orig = loaded_data["all_pairs_of_images_reduced_orig"] 
all_actions_one_hot = loaded_data["all_actions_one_hot"]
all_high_lvl_actions_one_hot = loaded_data["all_high_lvl_actions_one_hot"]
mean_all = loaded_data["mean_all"] 
std_all = loaded_data["std_all"] 
all_actions_unique = loaded_data["all_actions_unique"] 
orig_max = loaded_data["orig_max"] 
orig_min = loaded_data["orig_min"] 
#

train_set_no_dupp = loaded_data["train_set_no_dupp_processed"]
train_set_no_dupp_orig = loaded_data["train_set_no_dupp_orig"]




dico_highAction_lowLvlIds = {}

# loop over the train set (without dupplicate)
# AND group the transitions into their respective High Level Actions
for ii, ele in enumerate(train_set_no_dupp):


    # if the looped transition high level action is not a key of dico_highAction_lowLvlIds
    if np.argmax(ele[2]) not in dico_highAction_lowLvlIds:

        # add the key (the high lvl action index)
        dico_highAction_lowLvlIds[np.argmax(ele[2])] = []

    # if the hash is not in the keys of the dico of the high lvl action
    # i.e. if the transition was not added as a transition for this high lvl action
    if np.argmax(ele[1]) not in dico_highAction_lowLvlIds[np.argmax(ele[2])]:
        
        dico_highAction_lowLvlIds[np.argmax(ele[2])].append(np.argmax(ele[1]))

        

print(dico_highAction_lowLvlIds[0])




# print("dico_highAction_lowLvlIds")
# print(dico_highAction_lowLvlIds)



# BUT: chaque ligne = lowLvlId liste of effects

# on a high / lowLvlIds et high / list_of_effects (to keep)

#   donc


# 1) faire lowLvl / High 

dico_lowLvl_High = {}

for high, low_lvls in dico_highAction_lowLvlIds.items():
    for ll in low_lvls:
        if ll in dico_lowLvl_High.keys():
            print("PUTAIN DFE PROBLEM ")
            exit()
        else:
            dico_lowLvl_High[ll] = high


dico_lowLvl_High = {key: dico_lowLvl_High[key] for key in sorted(dico_lowLvl_High)}


with open("shap_vals_persisting_effects_removed/ALL_Transitions_And_HighLvlEffects.txt", "w") as file:

    #for eff in effects_to_be_taken:

    for llId, HighLvl in dico_lowLvl_High.items():

        #dico_lowLvl_HighLvlEffects[llId] = dico_highAction_filteredEffects[HighLvl]
        file.write(str(llId)+" ")
        file.write(",".join(dico_highAction_filteredEffects[HighLvl]))
        file.write("\n")

# print("dico_lowLvl_HighLvlEffectsdico_lowLvl_HighLvlEffects")
# print(dico_lowLvl_HighLvlEffects)
exit()
# 2) puis 
