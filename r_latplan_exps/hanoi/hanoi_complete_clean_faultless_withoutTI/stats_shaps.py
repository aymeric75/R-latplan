import statistics
import argparse
from collections import Counter

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process a type argument.")
parser.add_argument("--type", type=str, required=True, help="Specify the type to be processed.")

# Parse the arguments
args = parser.parse_args()

dic_shap_perEffect_perTrans_perAction = {}
last_key = ""




### BUILD THE DIC


for num_action in range(0, 22):

    dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)] = {}

    #with open("shap_vals_persisting_effects_removed/action_"+str(num_action)+".txt", "r") as file:
    with open("shap_vals_persisting_effects_removed/action_"+str(num_action)+"_withEmptySet.txt", "r") as file:

        for line in file:

            if "transition" in line:
                dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)][line.split(" ")[1].strip()] = {}
                last_key = line.split(" ")[1].strip()

            elif "add_" in line or "del_" in line:
                if len(dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)][last_key].values()) == 0:
                    dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)][last_key] = {}
                dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)][last_key][line.split(" ")[0].strip()] = float(line.split(" ")[1].strip())


# print("dic_shap_perEffect_perTrans_perActiondic_shap_perEffect_perTrans_perAction")

# print(dic_shap_perEffect_perTrans_perAction)

## NO 

if args.type == "stats":

    # for each action, write a file
    #
    #   where 1st line = nber of transitions
    #         2nd to 5th line = name_effect  nber_of_trans_in_which_present


    # for each action, retrieve all the transitions
    # (and go over the shap value of each effect of each transition)
    for action_name, transitions in dic_shap_perEffect_perTrans_perAction.items():

        nber_of_transitions = len(transitions.keys())

        all_effects = []

        effects_to_be_taken = []

        total_shap = 0 # total cumulated SHAP value of ALL effects of ALL transition of the ACTION

        # for each transition, and corresponding shaps (of each effect)
        for trans_id, effect_shap in transitions.items():

            all_effects.extend(effect_shap.keys())
            for val in effect_shap.values():
                total_shap += float(val)


        # print("all_effectsall_effects")
        # print(all_effects)

        print("nber_of_transitions {}".format(str(nber_of_transitions)))

        # count the number of each effect
        element_counts = Counter(all_effects)


        # Sort by most numerous first
        sorted_counts = element_counts.most_common()

        transitions_taken_care_of = []

        shap_value_covered_so_far = 0

        # le nbre de transitions que ça couvre
        # le % de SHAP value p.r.à SHAP total
        # Display the results
        # loop over EFFECTS by their SHAP importance (most important first)
        for element, count in sorted_counts:
            
            effects_to_be_taken.append(element)

            for trans_id, effect_shap in transitions.items():
                
                if element in effect_shap.keys():
                    if trans_id not in transitions_taken_care_of:
                        transitions_taken_care_of.append(trans_id)
                
                    
                    shap_value_covered_so_far += float(effect_shap[element])
                    # for val in effect_shap.values():
                    #     shap_value_covered_so_far += float(val)
        

            print(f"effect {element}: covers {count} transitions, total covered trans so far : {len(transitions_taken_care_of)}, shap so far: {'%.2f'%(shap_value_covered_so_far)}/{'%.2f'%(total_shap)}") 

            perc_covered = ( shap_value_covered_so_far / total_shap ) * 100

            #if len(transitions_taken_care_of) == nber_of_transitions and '%.1f'%(shap_value_covered_so_far) == '%.1f'%(total_shap):
            if len(transitions_taken_care_of) == nber_of_transitions and perc_covered > 60:
                    
                with open("shap_vals_persisting_effects_removed/"+str(action_name)+"_main_effects_withEmptySet_corrected.txt", "w") as file:
         
                    for eff in effects_to_be_taken:
                        file.write(eff+"\n")
                                
                break

        # 

        # print("LA LISTE FINALE")
        # print(effects_to_be_taken)

        # with open("shap_vals_persisting_effects_removed/"+action_name+"_STATS.txt", "w") as file:
        #     file.write(eff+"\n")





exit()

# Une fois la liste pour chaque action, si shap_value_covered_so_far ~= total_shap à une decimal près

#                   ET  si total covered so far EST LE même !!!


# ALORS c'est la liste des  EFFECTQS 




def Nmaxelements(list1, N):
    final_list = []

    for i in range(0, N):
        max1 = 0

        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j]

        list1.remove(max1)
        final_list.append(max1)

    return final_list





# and_set = []


# for key_trans, dico_shaps_for_trans in dic_shap_perEffect_perTrans_perAction.items():

#     effects = dico_shaps_for_trans.keys()

#     shap_vals = dico_shaps_for_trans.values()
#     mean_of_shaps_for_the_transition = statistics.mean(list(shap_vals))
    
#     print("list(shap_vals)")
#     print(list(shap_vals))

#     max_nth_elements = Nmaxelements(list(shap_vals), 1)
#     # print("max_nth_elements")
#     # print(max_nth_elements)
#     # exit()

#     for eff, shap in dico_shaps_for_trans.items():
#         if eff not in and_set:
#             # normal case
#             # and_set.append(eff)

#             # # case where we take only the positive shaps
#             # if shap > 0:
#             #     and_set.append(eff)

#             # # case where we take only the effects for which shap is > mean
#             # if shap > mean_of_shaps_for_the_transition:
#             #     and_set.append(eff)

#             # case where we take the Nth largest values
#             if shap in max_nth_elements:
#                 and_set.append(eff)


# print("and_set")
# print(and_set)
# print(len(and_set))



