#!/usr/bin/env python3

import os
import argparse
import sys
import subprocess

def switch_conda_environment(env_name):
    subprocess.run(f"conda activate {env_name}", shell=True)
    print(f"Switched to conda environment: {env_name}")


parser = argparse.ArgumentParser(description="A script to create the R-latplan datasets")
parser.add_argument('task', type=str, choices=['create_clean_traces', 'create_exp_data'], help='type of task to be performed')
parser.add_argument('domain', type=str, choices=['hanoi', 'blocks', 'sokoban'], help='domain name')
parser.add_argument('complete', type=str, choices=['complete', 'partial'], help='completness of the dataset, i.e. if missing transitions or not')
parser.add_argument('clean', type=str, choices=['clean', 'noisy'], help='if the dataset should be clean or noisy')
parser.add_argument('erroneous', type=str, choices=['erroneous', 'faultless'], help='if the dataset should contain some mislabeled transitions')


args = parser.parse_args()



##########   RULES   ############

# IF partial
#   
#       create X different datasets (each time with one node remove)

# 




##########   CREATE THE FOLDERS  ###########

### Create the "dataset" folder and the subfolder for this very dataset, if does not exist
if not os.path.exists("r_latplan_datasets"):
    os.makedirs("r_latplan_datasets") 

### Create the domain subfolder
if not os.path.exists("r_latplan_datasets/"+args.domain):
    os.makedirs("r_latplan_datasets/"+args.domain) 

## create the subfolder of a particular experiment
dataset_folder_name = args.domain+"_"+args.complete+"_"+args.clean+"_"+args.erroneous
if not os.path.exists("r_latplan_datasets/"+args.domain+"/"+dataset_folder_name):
    os.makedirs("r_latplan_datasets/"+args.domain+"/"+dataset_folder_name) 


exp_dir = os.getcwd()+'/'+"r_latplan_datasets/"+args.domain+"/"+dataset_folder_name
trace_dir = os.getcwd()+'/'+"r_latplan_datasets/"+args.domain

# ##########   CREATE THE TRACES OF CLEAN IMAGES  ###########

if args.task == "create_clean_traces":

    #switch_conda_environment("latplan")

    script_path = './r_latplan_datasets/pddlgym/pddlgym/genTraces.py'

    traces_dir = os.getcwd()+'/'+"r_latplan_datasets/"+args.domain
    args2 = [args.domain,  traces_dir]

    result = subprocess.run(['python', script_path] + args2, capture_output=False, text=True)

    #print(result.stdout)
    if result.stderr:
        print(result.stderr)


##########   FILL the EXP SUBFOLDER WITH  THE PAIRS of IMAGES and the LABEL for each PAIR  (and the init/goal stuff) ###########

if args.task == "create_exp_data":

    ### types of exp
    ###             partial / complete 
    ###
    ###             noisy / clean
    ###
    ###             noisy trans / not noisy trans


    if args.complete == "partial":

        args_dict = {
            "--domain": args.domain,
            "--exp_folder": exp_dir,
        }

        # Define the name of the Conda environment and the script with its arguments
        conda_env_name = 'graphviz'
        script_path = 'r_latplan_datasets/pddlgym/pddlgym/networkX_returnTransitionsToRemove.py'
        script_args = ["--domain "+str(args.domain), "--exp_folder "+str(exp_dir)]

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




    #from r_latplan_datasets.pddlgym.pddlgym import networkX_returnTransitionsToRemove
    #switch_conda_environment("latplan")

    #return_transitions_to_remove()

    script_path = './r_latplan_datasets/pddlgym/pddlgym/genDatasets.py'


    # Dictionary of arguments for genDatasets.py function
    complete_bool = None
    if args.complete == "complete":
        complete_bool = "True"
    elif args.complete == "partial":
        complete_bool =  "False"


    clean_bool = None
    if args.clean == "clean":
        clean_bool = "False"
    elif args.clean == "noisy":
        clean_bool =  "True"


    erroneous_bool = None
    if args.erroneous == "erroneous":
        erroneous_bool = "True"
    elif args.erroneous == "faultless":
        erroneous_bool =  "False"


    args_dict = {
        "--trace_dir": trace_dir,
        "--exp_folder": exp_dir,
        "--domain": args.domain,
        "--remove_some_trans": complete_bool,
        "--add_noisy_trans" : clean_bool,
        "--ten_percent_noisy_and_dupplicated": erroneous_bool
    }

    # Convert the dictionary to a list of arguments
    args_list = []
    for key, value in args_dict.items():
        args_list.append(key)
        if value is not None:
            args_list.append(value)
    
    result = subprocess.run(['python', script_path] + args_list, capture_output=False, text=True)








### Proceed on, First, creating the traces from PDDLGym (i.e. traces of Pairs of images + transition labels)

### 