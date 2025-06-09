#!/usr/bin/env python3

import os
import argparse
import sys
import subprocess

def switch_conda_environment(env_name):
    subprocess.run(f"conda activate {env_name}", shell=True)
    print(f"Switched to conda environment: {env_name}")


parser = argparse.ArgumentParser(description="A script to train R-latplan for a specific experiment")
parser.add_argument('type', type=str, choices=['r_latplan', 'vanilla'], help='if vanilla or r-latplan')
parser.add_argument('domain', type=str, choices=['hanoi', 'blocks', 'sokoban'], help='domain name')
parser.add_argument('dataset_folder', type=str, help='folder where the images are')
parser.add_argument('--pb_folder', default="", type=str, help='REQUIRED for PARTIAL', required=False)

args = parser.parse_args()

if 'partial' in args.dataset_folder and args.pb_folder == "":
    print("pb_folder ARG is REQUIRED")
    exit()





##########   RULES   ############

# IF partial
#   
#       create X different datasets (each time with one node remove)
# 





dico_blocks = {
    "task": "blocks",
    "type": "cylinders-4-flat",
    "width_height": "",
    "nb_examples": "20000",
    "conf_folder": ""

}


dico_sokoban = {
    "task": "sokoban",
    "type": "sokoban_image-20000-global-global-2-train",
    "width_height": "",
    "nb_examples": "20000",
    "conf_folder": ""
}

dico_hanoi = {
    "task": "hanoi",
    "type": "",
    "width_height": "4 4",
    "nb_examples": "5000",
    "conf_folder": ""
}




## create the subfolder of a particular experiment

_exp_base = None

if args.type == "r_latplan":
    _exp_base = "r_latplan_exps"

elif args.type == "vanilla":
    _exp_base = "r_vanilla_latplan_exps"  


if not os.path.exists(_exp_base + "/"+args.domain+"/"+args.dataset_folder):
    os.makedirs(_exp_base + "/"+args.domain+"/"+args.dataset_folder) 

exp_folder = _exp_base + "/"+args.domain+"/"+args.dataset_folder



dico_ = None

if args.domain == "sokoban":
    dico_ = dico_sokoban
elif args.domain == "blocks":
    dico_ = dico_blocks
elif args.domain == "hanoi":
    dico_ = dico_hanoi


script_path = './train_kltune.py'
# learn sokoban sokoban_image-20000-global-global-2-train  20000 CubeSpaceAE_AMA4Conv kltune2 --hash NoisyPartialDFA



if 'partial' in args.dataset_folder:
    dataset_fold = args.dataset_folder+"/pbs/" + args.pb_folder
else:
    dataset_fold = args.dataset_folder


#exit()

args_dict = {
    "learn": None,
    dico_["task"]: None,
    dico_["type"]: None,
    dico_["width_height"]: None,
    dico_["nb_examples"]: None,
    "CubeSpaceAE_AMA4Conv": None,
    "kltune": None,
    "--dataset_folder": dataset_fold,
    "--type": args.type,
}

# Convert the dictionary to a list of arguments
args_list = []
args_list_str = ""
for key, value in args_dict.items():
    args_list.append(key)
    args_list_str += str(key)+" "
    if value is not None:
        args_list.append(value)
        args_list_str += str(value)+" "


# learn hanoi  4 4 5000 CubeSpaceAE_AMA4Conv kltune2 --hash NoisyPartialDFA2


outfile_str = "/workspace/R-latplan/"+_exp_base+"/" + args.domain  + "/" + dataset_fold + "/fileTEST.out"
errfile_str = "/workspace/R-latplan/"+_exp_base+"/" + args.domain  + "/" + dataset_fold + "/fileTEST.err"

# print("errfile_str")
# print(errfile_str)

# print("outfile_str")
# print(outfile_str)

# exit()

#result = subprocess.run(['python', script_path] + args_list, capture_output=False, text=True)


with open(outfile_str, "w") as outfile, open(errfile_str, "w") as errfile:

    result = subprocess.run('/workspace/R-latplan/train_kltune.py ' + args_list_str, shell = True, check = True, capture_output=False, stdout = outfile, stderr = errfile)
    #result = subprocess.run(['python','/workspace/R-latplan/train_kltune.py ' + args_list_str], shell = True, check = False, capture_output = True)

