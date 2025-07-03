#!/usr/bin/env python3

import os
import argparse
import sys
import subprocess
import shutil
import numpy as np
import random
import copy

def switch_conda_environment(env_name):
    subprocess.run(f"conda activate {env_name}", shell=True)
    print(f"Switched to conda environment: {env_name}")


parser = argparse.ArgumentParser(description="A script to create the R-latplan datasets")
parser.add_argument('type', type=str, choices=['r_latplan', 'vanilla'], help='if vanilla or r-latplan')
parser.add_argument('task', type=str, choices=['cut', 'create_clean_traces', 'create_exp_data_sym', 'create_exp_data_im'], help='type of task to be performed')
parser.add_argument('domain', type=str, choices=['hanoi', 'blocks', 'sokoban'], help='domain name')



parser.add_argument('complete', type=str, choices=['complete', 'partial'], help='completness of the dataset, i.e. if missing transitions or not')
parser.add_argument('clean', type=str, choices=['clean', 'noisy'], help='if the dataset should be clean or noisy')
parser.add_argument('erroneous', type=str, choices=['erroneous', 'faultless'], help='if the dataset should contain some mislabeled transitions')
parser.add_argument('use_transi_iden', type=str, choices=["True", "False"], help='if to use the transition identifier')
parser.add_argument('extra_label', type=str, help='add an extra label to the repo', default="NOT")


args = parser.parse_args()



use_transi_iden = False
if args.use_transi_iden == "True" and args.type == "r_latplan":
    use_transi_iden = True


def bool_to_str(s):
    if s:
        return "True"
    return "False"

## ALL EXPS:
##      Exp1: complete clean faultless
##      Exp2: complete noisy faultless
##      Exp3: partial clean faultless
##      Exp4: partial noisy faultless
##      Exp5: complete noisy erroneous ("symbols" are same as Exp2)
##      




# Booleans for decinding the type of experiments

complete_bool = None
if args.complete == "complete":
    complete_bool = True
elif args.complete == "partial":
    complete_bool =  False

clean_bool = None
if args.clean == "clean":
    clean_bool = True
elif args.clean == "noisy":
    clean_bool =  False

erroneous_bool = None
if args.erroneous == "erroneous":
    erroneous_bool = True
elif args.erroneous == "faultless":
    erroneous_bool =  False




##########   CREATE THE FOLDERS and SUBFOLDERS   ###########

### Create the "dataset" folder and the subfolder for this very dataset, if does not exist
if args.type == "r_latplan":

    if not os.path.exists("r_latplan_datasets"):
        os.makedirs("r_latplan_datasets") 
    ### Create the domain subfolder
    if not os.path.exists("r_latplan_datasets/"+args.domain):
        os.makedirs("r_latplan_datasets/"+args.domain) 


    use_ti = "withoutTI"
    if use_transi_iden == True:
        use_ti = "withTI"

    ## create the subfolder of a particular experiment
    dataset_folder_name = args.domain+"_"+args.complete+"_"+args.clean+"_"+args.erroneous + "_" + use_ti
    if args.extra_label != "NOT":
        dataset_folder_name = dataset_folder_name + "_"  + args.extra_label

    if not os.path.exists("r_latplan_datasets/"+args.domain+"/"+dataset_folder_name):
        os.makedirs("r_latplan_datasets/"+args.domain+"/"+dataset_folder_name) 

    dataset_exp_dir = os.getcwd()+'/'+"r_latplan_datasets/"+args.domain+"/"+dataset_folder_name

    exp_exp_dir = os.getcwd()+'/'+"r_latplan_exps/"+args.domain+"/"+dataset_folder_name

    trace_dir = os.getcwd()+'/'+"r_latplan_datasets/"+args.domain


elif args.type == "vanilla":

    if not os.path.exists("r_vanilla_latplan_datasets"):
        os.makedirs("r_vanilla_latplan_datasets") 
    ### Create the domain subfolder
    if not os.path.exists("r_vanilla_latplan_datasets/"+args.domain):
        os.makedirs("r_vanilla_latplan_datasets/"+args.domain) 

    ## create the subfolder of a particular experiment
    dataset_folder_name = args.domain+"_"+args.complete+"_"+args.clean+"_"+args.erroneous
    if args.extra_label != "NOT":
        dataset_folder_name = dataset_folder_name + "_"  + args.extra_label
    if not os.path.exists("r_vanilla_latplan_datasets/"+args.domain+"/"+dataset_folder_name):
        os.makedirs("r_vanilla_latplan_datasets/"+args.domain+"/"+dataset_folder_name) 

    dataset_exp_dir = os.getcwd()+'/'+"r_vanilla_latplan_datasets/"+args.domain+"/"+dataset_folder_name

    exp_exp_dir = os.getcwd()+'/'+"r_vanilla_latplan_exps/"+args.domain+"/"+dataset_folder_name

    trace_dir = os.getcwd()+'/'+"r_vanilla_latplan_datasets/"+args.domain


import pickle
def load_dataset(path_to_file):
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data


if args.task == "cut":



    
    dataset_exp_dir = dataset_exp_dir + "_N25"
    
    exp_exp_dir = exp_exp_dir + "_N25"

    print("dataset_exp_dir") # /workspace/R-latplan/r_latplan_datasets/hanoi/hanoi_complete_clean_faultless_withoutTI
    print(dataset_exp_dir)
    print("exp_exp_dir") # /workspace/R-latplan/r_latplan_exps/hanoi/hanoi_complete_clean_faultless_withoutTI
    print(exp_exp_dir)


    print("trace_dir") # /workspace/R-latplan/r_latplan_datasets/hanoi
    print(trace_dir)

    # 0) locate the data.p (if not seen, return)
    if not os.path.exists(dataset_exp_dir + "/" + "data.p"):
        
        print("ffffffffff")
        print(dataset_exp_dir)
        print("File DOES NOT exists")
        exit()
    

    # 1) load it and remove X% of the transitions
    loaded_data = load_dataset(dataset_exp_dir + "/" + "data.p")

    # train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, all_high_lvl_actions_one_hot, mean_all, std_all, 
    # all_actions_unique, all_high_lvl_actions_unique, orig_max, orig_min, train_set_no_dupp_processed, train_set_no_dupp_orig, all_traces_pair_and_action

    

    ## 1) i) 
    train_set = loaded_data["train_set"] 
    test_val_set = loaded_data["test_val_set"] 
    all_pairs_of_images_reduced_orig = loaded_data["all_pairs_of_images_reduced_orig"] 
    all_actions_one_hot = loaded_data["all_actions_one_hot"]
    all_high_lvl_actions_one_hot = loaded_data["all_high_lvl_actions_one_hot"] 
    mean_all = loaded_data["mean_all"] 
    std_all = loaded_data["std_all"]
    all_actions_unique = loaded_data["all_actions_unique"] 
    all_high_lvl_actions_unique = loaded_data["all_high_lvl_actions_unique"]
    orig_max = loaded_data["orig_max"]
    orig_min = loaded_data["orig_min"]
    train_set_no_dupp_processed = loaded_data["train_set_no_dupp_processed"]
    train_set_no_dupp_orig = loaded_data["train_set_no_dupp_orig"]
    all_traces_pair_and_action = loaded_data["all_traces_pair_and_action"]
    
    


    #### Compute the number of llps per lla
    ##### ENSUITE retire X% de chaque lla

    # GROUP THE TRANSITIONS BY THEIR HIGH LVL ACTION
    dico_transitions_per_high_lvl_actions = {}
    for ii, ele in enumerate(train_set_no_dupp_processed):
        if np.argmax(ele[2]) not in dico_transitions_per_high_lvl_actions:
            dico_transitions_per_high_lvl_actions[np.argmax(ele[2])] = []
        if np.argmax(ele[1]) not in dico_transitions_per_high_lvl_actions[np.argmax(ele[2])]:
            dico_transitions_per_high_lvl_actions[np.argmax(ele[2])].append(np.argmax(ele[1]))



    # 
    transis_to_remove = []

    for h_in, high in dico_transitions_per_high_lvl_actions.items():

        size_percented = len(high) // 5

        to_remove = random.sample(high, size_percented)

        for el in to_remove:
            transis_to_remove.append(el)

    print("transis_to_remove")
    print(len(np.array(transis_to_remove)))

    print(len(np.unique(np.array(transis_to_remove))))


    # 2) enlever from # remove from train_set, test_val_set, all_actions_unique, train_set_no_dupp_processed, train_set_no_dupp_orig

    #
    # print(type(train_set)) # <class 'list'>
    # print(type(test_val_set)) # <class 'list'>
    # print(type(all_actions_unique)) # <class 'list'>
    # print(type(train_set_no_dupp_processed)) # <class 'list'>
    # print(type(train_set_no_dupp_orig)) # <class 'list'>


    # 
    train_set_ = copy.deepcopy(train_set)
    test_val_set_ = copy.deepcopy(test_val_set)
    all_actions_unique_ = copy.deepcopy(all_actions_unique)
    train_set_no_dupp_processed_ = copy.deepcopy(train_set_no_dupp_processed)
    train_set_no_dupp_orig_ = copy.deepcopy(train_set_no_dupp_orig)

    all_actions_one_hot_ = copy.deepcopy(all_actions_one_hot)


    for i, ele in enumerate(train_set):
        if np.argmax(ele[-2]) in transis_to_remove:
            train_set_[i] = None
        else:
            train_set_[i][-2] = np.delete(train_set_[i][-2], transis_to_remove)

    train_set_ = [x for x in train_set_ if x is not None]

    for i, ele in enumerate(test_val_set):
        if np.argmax(ele[-2]) in transis_to_remove:
            test_val_set_[i] = None
        else:
            test_val_set_[i][-2] = np.delete(test_val_set_[i][-2], transis_to_remove)
    test_val_set_ = [x for x in test_val_set_ if x is not None]


    for i, ele in enumerate(all_actions_unique):
        if i in transis_to_remove:
            all_actions_unique_[i] = None
    all_actions_unique_ = [x for x in all_actions_unique_ if x is not None]


    for i, ele in enumerate(train_set_no_dupp_processed):
        if np.argmax(ele[-2]) in transis_to_remove:
            train_set_no_dupp_processed_[i] = None
        else:
            train_set_no_dupp_processed_[i][-2] = np.delete(train_set_no_dupp_processed_[i][-2], transis_to_remove)
    train_set_no_dupp_processed_ = [x for x in train_set_no_dupp_processed_ if x is not None]


    for i, ele in enumerate(train_set_no_dupp_orig):
        if np.argmax(ele[-2]) in transis_to_remove:
            train_set_no_dupp_orig_[i] = None
        else:
            train_set_no_dupp_orig_[i][-2] = np.delete(train_set_no_dupp_orig_[i][-2], transis_to_remove)
    train_set_no_dupp_orig_ = [x for x in train_set_no_dupp_orig_ if x is not None]

    for i, ele in enumerate(all_actions_one_hot):
        if np.argmax(ele) in transis_to_remove:
            all_actions_one_hot_[i] = None
    all_actions_one_hot_ = [x for x in all_actions_one_hot_ if x is not None]

    data = {
        "train_set": train_set_,
        "test_val_set": test_val_set_,
        "all_pairs_of_images_reduced_orig": all_pairs_of_images_reduced_orig,
        "all_actions_one_hot": all_actions_one_hot_,
        "all_high_lvl_actions_one_hot": all_high_lvl_actions_one_hot,
        "mean_all": mean_all,
        "std_all": std_all,
        "all_actions_unique": all_actions_unique_, 
        "all_high_lvl_actions_unique": all_high_lvl_actions_unique, 
        "orig_max": orig_max, 
        "orig_min": orig_min, 
        "train_set_no_dupp_processed": train_set_no_dupp_processed_, 
        "train_set_no_dupp_orig": train_set_no_dupp_orig_, 
        "all_traces_pair_and_action": all_traces_pair_and_action
    }


    filename = "dataPartialLast.p"
    with open(exp_exp_dir+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)
    with open(dataset_exp_dir+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)

    # dataset_exp_dir  exp_exp_dir

    exit()

    # dataPartialLast.p

    # OK, donc:

    # 1) training

    # 2) 



    
    #
    # save it back on the right folder (should be in r_latplan_datasets)
    #

##########   CREATE THE TRACES OF CLEAN IMAGES  ###########

if args.task == "create_clean_traces":

    #switch_conda_environment("latplan")

    script_path = './r_latplan_datasets/pddlgym/pddlgym/genTraces.py'

    traces_dir = "traceHanoi"

    if args.type == "r_latplan":
        traces_dir = "r_latplan_exps/"+args.domain
    else:
        traces_dir = "r_vanilla_latplan_exps/"+args.domain


    

    args2 = [args.domain,  traces_dir]

    result = subprocess.run(['python', script_path] + args2, capture_output=False, text=True)

    #print(result.stdout)
    if result.stderr:
        print(result.stderr)


##########   FILL the EXP SUBFOLDER WITH  THE PAIRS of IMAGES and the LABEL for each PAIR  (and the init/goal stuff) ###########

if args.task == "create_exp_data_sym":


    ####################### CREATE THE SYMBOLIC PROBLEMS DATA ###################################


    if complete_bool == True:

        script_path_2 = './r_latplan_datasets/pddlgym/pddlgym/networkX-genGraph.py'

        # Define the name of the Conda environment and the script with its arguments
        conda_env_name = 'graphviz'

        script_args = ["--domain "+str(args.domain), "--exp_folder "+str(dataset_exp_dir)]

        command = f'''
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate {conda_env_name}
        python {script_path_2} {" ".join(script_args)}
        '''
        # Use subprocess to run the command in a shell
        process = subprocess.Popen(command, shell=True, executable='/bin/bash')
        process.communicate()

    else:

        args_dict = {
            "--domain": args.domain,
            "--exp_folder": dataset_exp_dir,
        }

        # Define the name of the Conda environment and the script with its arguments
        conda_env_name = 'graphviz'
        script_path = 'r_latplan_datasets/pddlgym/pddlgym/networkX_returnTransitionsToRemove.py'
        script_args = ["--domain "+str(args.domain), "--exp_folder "+str(dataset_exp_dir)]

        # Construct the command to activate the Conda environment and run the script
        #command = f'conda activate {conda_env_name} && python {script_path} {" ".join(script_args)}'

        command = f'''
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate {conda_env_name}
        python {script_path} {" ".join(script_args)}
        '''

        # Use subprocess to run the command in a shell
        process = subprocess.Popen(command, shell=True, executable='/bin/bash')
        process.communicate()


if args.task == "create_exp_data_im":

    ############ CREATE THE IMAGES (pairs and init/goal) ###################

    #from r_latplan_datasets.pddlgym.pddlgym import networkX_returnTransitionsToRemove
    #switch_conda_environment("latplan")

    #return_transitions_to_remove()

    script_path = './r_latplan_datasets/pddlgym/pddlgym/genDatasets.py'

    use_transi_iden_str = "False"
    if use_transi_iden:
        use_transi_iden_str = "True"


    args_dict = {
        "--trace_dir": trace_dir,
        "--exp_folder": dataset_exp_dir,
        "--domain": args.domain,
        "--remove_some_trans": not complete_bool,
        "--add_noisy_trans" : not clean_bool,
        "--ten_percent_noisy_and_dupplicated": erroneous_bool,
        "--type_exp": args.type,
        "--use_transi_iden": use_transi_iden_str,
        
    }

    # Convert the dictionary to a list of arguments


    if args.complete == 'partial':
        subfolders = [f.name for f in os.scandir(dataset_exp_dir + "/pbs") if f.is_dir()]
        for subf in subfolders:

            args_list = []
            for key, value in args_dict.items():
                args_list.append(key)
                if value is not None:
                    args_list.append(str(value))
            
            args_list.append("--pb_folder")
            args_list.append(subf)

            
            print(args_list)
            result = subprocess.run(['python', script_path] + args_list, capture_output=False, text=True)
    
    # call genDatasets.py for each of the pbs folders present in the pbs folder
    # 
    
    else:
        
        args_list = []
        for key, value in args_dict.items():
            args_list.append(key)
            if value is not None:
                args_list.append(str(value))
        

        result = subprocess.run(['python', script_path] + args_list, capture_output=False, text=True)


    # COPY PASTE THE PBS folder into exp_exp_dir
    

    import os
    import shutil

    source_dir = os.path.join(dataset_exp_dir, "pbs")
    dest_dir = os.path.join(exp_exp_dir, "pbs")


    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, dirs, files in os.walk(source_dir):
        rel_path = os.path.relpath(root, source_dir)
        dest_path = os.path.join(dest_dir, rel_path)

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_path, file)
            
            if not os.path.exists(dest_file):
                shutil.copy2(src_file, dest_file)

        for dir in dirs:
            src_dir = os.path.join(root, dir)
            dest_subdir = os.path.join(dest_path, dir)
            
            if not os.path.exists(dest_subdir):
                shutil.copytree(src_dir, dest_subdir)



### Proceed on, First, creating the traces from PDDLGym (i.e. traces of Pairs of images + transition labels)

### 