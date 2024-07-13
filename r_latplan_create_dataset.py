#!/usr/bin/env python3

import os
import argparse
import sys
import subprocess
import shutil

def switch_conda_environment(env_name):
    subprocess.run(f"conda activate {env_name}", shell=True)
    print(f"Switched to conda environment: {env_name}")


parser = argparse.ArgumentParser(description="A script to create the R-latplan datasets")
parser.add_argument('type', type=str, choices=['r_latplan', 'vanilla'], help='if vanilla or r-latplan')
parser.add_argument('task', type=str, choices=['create_clean_traces', 'create_exp_data_sym', 'create_exp_data_im'], help='type of task to be performed')
parser.add_argument('domain', type=str, choices=['hanoi', 'blocks', 'sokoban'], help='domain name')
parser.add_argument('complete', type=str, choices=['complete', 'partial'], help='completness of the dataset, i.e. if missing transitions or not')
parser.add_argument('clean', type=str, choices=['clean', 'noisy'], help='if the dataset should be clean or noisy')
parser.add_argument('erroneous', type=str, choices=['erroneous', 'faultless'], help='if the dataset should contain some mislabeled transitions')
parser.add_argument('use_transi_iden', type=str, choices=["True", "False"], help='if to use the transition identifier')
parser.add_argument('extra_label', type=str, help='add an extra label to the repo', default="NOT")


args = parser.parse_args()


use_transi_iden = False
if args.use_transi_iden == "True":
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






##########   CREATE THE TRACES OF CLEAN IMAGES  ###########

if args.task == "create_clean_traces":

    #switch_conda_environment("latplan")

    script_path = './r_latplan_datasets/pddlgym/pddlgym/genTraces.py'

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

    args_dict = {
        "--trace_dir": trace_dir,
        "--exp_folder": dataset_exp_dir,
        "--domain": args.domain,
        "--remove_some_trans": not complete_bool,
        "--add_noisy_trans" : not clean_bool,
        "--ten_percent_noisy_and_dupplicated": erroneous_bool,
        "--type_exp": args.type,
        "--use_transi_iden": args.use_transi_iden,
        
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

    exit()
    # COPY PASTE THE PBS folder into exp_exp_dir
    

    if not os.path.exists(exp_exp_dir+"/pbs"):
        os.makedirs(exp_exp_dir+"/pbs")  

    shutil.copytree(dataset_exp_dir+"/pbs", exp_exp_dir+"/pbs", dirs_exist_ok=True)



### Proceed on, First, creating the traces from PDDLGym (i.e. traces of Pairs of images + transition labels)

### 