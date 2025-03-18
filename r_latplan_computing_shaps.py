#!/usr/bin/env python3

import os
import argparse
import sys
import subprocess
import re 
import shutil

def switch_conda_environment(env_name):
    subprocess.run(f"conda activate {env_name}", shell=True)
    print(f"Switched to conda environment: {env_name}")

def find_lowest_integer(directory):
    # Define the pattern to match the files
    pattern = re.compile(r'net0-\d+-(\d+)\.h5')
    
    # Initialize a variable to keep track of the lowest integer
    lowest_integer = None

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Extract the integer from the filename
            current_integer = int(match.group(1))
            
            # Update the lowest integer if necessary
            if lowest_integer is None or current_integer < lowest_integer:
                lowest_integer = current_integer
    
    return lowest_integer



def copy_and_rename_files(directory, specific_number, destination_directory):
    # Define the pattern to match the files containing the specific number
    pattern = re.compile(rf'(-\d+-){specific_number}(\.)')

    # # Ensure the destination directory exists
    # if not os.path.exists(destination_directory):
    #     os.makedirs(destination_directory)

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        match = pattern.search(filename)
        if match:
            # Construct the new filename by removing the specific number and the part matching -*
            new_filename = re.sub(r'-\d+-\d+', '', filename)
            
            # Copy the file to the destination directory with the new filename
            src_file = os.path.join(directory, filename)
            dst_file = os.path.join(destination_directory, new_filename)
            shutil.copyfile(src_file, dst_file)
            print(f'Copied {src_file} to {dst_file}')







parser = argparse.ArgumentParser(description="A script to test R-latplan for a specific experiment")
parser.add_argument('type', type=str, choices=['r_latplan', 'vanilla'], help='type of task to be performed')
parser.add_argument('task', type=str, choices=['aggregate_effects_preconds'], help='type of task to be performed')
parser.add_argument('domain', type=str, choices=['hanoi', 'blocks', 'sokoban'], help='domain name')
parser.add_argument('dataset_folder', type=str, help='folder where the PDDLs are')
parser.add_argument('--pb_folder', default="", type=str, help='REQUIRED for PARTIAL', required=False)

args = parser.parse_args()

if 'partial' in args.dataset_folder and args.pb_folder == "":
    print("pb_folder ARG is REQUIRED")
    exit()



###

if args.type == "r_latplan":
    exp_folder = "r_latplan_exps/"+args.domain+"/"+args.dataset_folder
else:
    exp_folder = "r_vanilla_latplan_exps/"+args.domain+"/"+args.dataset_folder
print("exp_folder")
print(exp_folder)


if args.task == "aggregate_effects_preconds":
    

    #####  retrieve all the low lvl actions for high high level action


    ###### generate the CSVs

    script_path = './train_kltune.py'

    if args.domain == "hanoi":
        task="hanoi"
        typee=""
        width="4"
        height="4"
        nb_examples="20000"

    elif args.domain == "blocks":
        task="blocks"
        typee="cylinders-4-flat"
        width_height=""
        nb_examples="20000"

    elif args.domain == "sokoban":
        task="sokoban"
        typee="sokoban_image-20000-global-global-2-train"
        width=""
        height=""
        nb_examples="20000"

    #$type $width_height $nb_examples CubeSpaceAE_AMA4Conv kltune2

    args_dict = {
        "dump": None,
        task: None,
        typee: None,
        "width": width,
        "height": height,
        nb_examples: None,
        "CubeSpaceAE_AMA4Conv" : None,
        "kltune2": None,
        "--dataset_folder": args.dataset_folder,
        "--type": args.type
    }


    # Convert the dictionary to a list of arguments
    args_list = []
    for key, value in args_dict.items():
        if ( key == "width" or key == "height" ):
            if args.domain != "sokoban":
                if str(value) != "":
                    args_list.append(str(value))
        else:
            if str(key) != "":
                args_list.append(key)
            if value is not None:
                if str(value) != "":
                    args_list.append(str(value))


    result = subprocess.run(['python', script_path] + args_list, capture_output=False, text=True)


    # args_list = [exp_folder]

    # script_path = './pddl-ama3.sh'

    # result = subprocess.run(['bash', script_path] + args_list, capture_output=False, text=True)

