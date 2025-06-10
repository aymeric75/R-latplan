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

    # 256 = 7 iterations


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
parser.add_argument('task', type=str, choices=['generate_pddl', 'transform_into_clustered_pddl', 'cluster_llas', 'gen_plans', 'gen_dfa_array', 'compute_effects_shap_values_per_action', 'compute_pos_preconds_shap_values_per_action', 'compute_neg_preconds_shap_values_per_action', 'gen_invariants'], help='type of task to be performed')
parser.add_argument('domain', type=str, choices=['hanoi', 'blocks', 'sokoban'], help='domain name')
parser.add_argument('dataset_folder', type=str, help='folder where the images are')
parser.add_argument('--pb_folder', default="", type=str, help='REQUIRED for PARTIAL', required=False)
parser.add_argument('--use_base_to_load', default=None, type=str, help='Optional: Base path to load data from', required=False)

parser.add_argument('--clustering_with_penalty', default=False, type =lambda x: x.lower() == 'true', help='Optional: indicates if we use or not the penalty in Jaccard distance', required=False)
parser.add_argument('--clustering_base_data', default=None, type=str, choices=['only_preconds', 'only_effects', 'both'], help='Optional: indicates which type of data are used for the clustering', required=False)


parser.add_argument('--specific_whens', default=False, type =lambda x: x.lower() == 'true', help='Optional: indicates whether to use specific whens or not', required=False)

parser.add_argument('--ors_in_whens_are_groups', default=False, type =lambda x: x.lower() == 'true', help='Optional: indicates if we use groups of literals (<=> pruned preconds) or just the union', required=False)



args = parser.parse_args()

if 'partial' in args.dataset_folder and args.pb_folder == "":
    print("pb_folder ARG is REQUIRED") 
    exit()




def bool_to_str(s):
    print(type(s))
    print(s)
    if s:
        return "True"
    return "False"

###

if args.type == "r_latplan":
    exp_folder = "r_latplan_exps/"+args.domain+"/"+args.dataset_folder
    data_folder  = "r_latplan_datasets/"+args.domain+"/"+args.dataset_folder

else:
    exp_folder = "r_vanilla_latplan_exps/"+args.domain+"/"+args.dataset_folder
    data_folder  = "r_vanilla_latplan_datasets/"+args.domain+"/"+args.dataset_folder



# compute_effects_shap_values_per_action or compute_preconds_shap_values_per_action
if args.task == "compute_effects_shap_values_per_action" or args.task == "compute_pos_preconds_shap_values_per_action" or args.task == "compute_neg_preconds_shap_values_per_action":


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
    args_dict = {}

    args_dict[args.task] = None
    args_dict[task] = None
    args_dict[typee] = None
    args_dict["width"] = width
    args_dict["height"] = height
    args_dict[nb_examples] = None
    args_dict["CubeSpaceAE_AMA4Conv" ] = None
    args_dict["kltune2"] = None
    args_dict["--dataset_folder"] = args.dataset_folder
    args_dict["--type"] = args.type



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

    exit()


def find_first_pddl_file(root_folder):
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".pddl"):
                return os.path.join(subdir, file)
    return None  # No .pddl file found

if args.task == "gen_invariants":

    # 1) récupère le domain et un problem (depuis pbs)
    domain_file = exp_folder+"/domainCondBIS_NORMAL.pddl" #/domain.pddl"
    if not os.path.exists(domain_file):
        print("domain file does not exists")
        exit()

    problem_file = find_first_pddl_file(exp_folder+"/pbs/")

    if problem_file:
        print(f"First .pddl file found: {problem_file}")
    else:
        print("No .pddl file found")

    # 2) "va dans" src/translate' et call transte.py

    script_path = "downward/src/translate/translate.py"


    # Convert the dictionary to a list of arguments
    # , "--sas-file "+exp_folder+"/output.sas"
    # 
    #args_list = [domain_file, problem_file, "--sas-file" ]


    args_dict = {
        domain_file: None,
        problem_file: None,
        "--sas-file": exp_folder+"/output.sas",

    }


    # Convert the dictionary to a list of arguments
    args_list = []
    for key, value in args_dict.items():

        if str(key) != "":
            args_list.append(key)
        if value is not None:
            if str(value) != "":
                args_list.append(str(value))

                
    result = subprocess.run(['python', script_path] + args_list, capture_output=False, text=True)

    #, "--no_bw_h2"

    with open(exp_folder+"/output.sas", "r") as infile:
        process = subprocess.run(
            ["/workspace/R-latplan/h2-preprocessor/builds/release32/bin/preprocess"],
            stdin=infile,  # Pass the file object, not a string
            text=True,  # Ensures input and output are treated as text
            capture_output=True  # Capture stdout and stderr
        )




    # process = subprocess.run(
    #     ["/workspace/R-latplan/h2-preprocessor/builds/release32/bin/preprocess", "--no_bw_h2"],
    #     stdin=exp_folder+"/output.sas",
    #     text=True,  # Ensures input and output are treated as text
    #     capture_output=False  # Captures stdout and stderr
    # )

    # # Print or process the output
    # print("STDOUT:", process.stdout)
    # print("STDERR:", process.stderr)





    # def mutexes_from_sas(sas_file):
            
    #     import re
    #     import sys

    #     START_PATTERN = '^begin_variable$'
    #     END_PATTERN = '^end_variable$'

    #     dict_vars={}
    #     with open(sas_file) as file:

    #         cc=0 # counter for within the variable
    #         between = False
            
    #         tmp_array=[]
    #         #counter_var=0
    #         key=0

    #         for line in file:

    #             if re.match(START_PATTERN, line):
    #                 cc=0
    #                 tmp_array = []
    #                 between = True
    #                 continue
    #             elif re.match(END_PATTERN, line):
    #                 between = False
    #                 if(len(tmp_array)>0):
    #                     dict_vars[key] = tmp_array
    #                 cc=0
    #                 continue
    #             else:
    #                 if(between == True):

    #                     if(cc==0):
                            
    #                         thenumber = re.findall(r'\d+', line)
    #                         key=int(thenumber[0])
                            
    #                     if(cc>2):
    #                         thenumber = re.findall(r'\d+', line)
    #                         if len(thenumber) > 0:
    #                             tmp_array.append(str(int(thenumber[0])))
    #                         if "none of those" in line:
    #                             tmp_array.append("NoneOfThose")
    #                     cc+=1
    #     file.close()


    #     START_PATTERN = '^begin_mutex_group$'
    #     END_PATTERN = '^end_mutex_group$'

    #     arr_mutexes=[]
    #     with open(sas_file) as file:

    #         cc=0 # counter for within the variable
    #         between = False
    #         tmp_array=[]
    #         key=0

    #         jj=0
    #         for line in file:

    #             if re.match(START_PATTERN, line):
    #                 cc=0
    #                 tmp_array = []
    #                 between = True
    #                 continue
    #             elif re.match(END_PATTERN, line):
    #                 between = False
    #                 if(len(tmp_array)>0):
    #                     arr_mutexes.append(tmp_array)
    #                 cc=0
    #                 continue
    #             else:
    #                 if(between == True):

    #                     if(cc>0):
    #                         #print(line)
    #                         var_number, value_index  = line.split(" ")
                            
    #                         z_index = dict_vars[int(var_number)][int(value_index)]

    #                         if "none of those" in str(z_index):
    #                             print("NONE in a MUTEX !!!!!!!!!!!!!")
    #                             exit()
    #                         else:
    #                             tmp_array.append(z_index)
    #                     cc+=1
    #             jj+=1


    #     file.close()

    #     # retourne variables, and the mutexes
    #     return dict_vars, arr_mutexes



    # dict_vars, arr_mutexes = mutexes_from_sas(exp_folder+"/output.sas")
    # result = subprocess.run(['bash', "./h2-preprocessor/builds/release32/bin"] + args_list, capture_output=False, text=True)



    # mutexes_to_send = []

    # # adding eles from dict_vars
    # for k, m in dict_vars.items():  
    #     if not (len(m) == 2 and m[0] == m[1]) and m not in mutexes_to_send:
    #         mutexes_to_send.append(m)

    # # adding eles from arr_mutexes
    # for ele in arr_mutexes:
    #     if ele not in mutexes_to_send:
    #         mutexes_to_send.append(ele)

    # with open(exp_folder+"/mutexes.txt", "w") as f:
    #     for mu in mutexes_to_send:
    #         f.write(mu+"\n")

    # args_list = [exp_folder]

    # script_path = './pddl-ama3.sh'

    # result = subprocess.run(['bash', script_path] + args_list, capture_output=False, text=True)



    
    exit()





if args.task == "generate_pddl":
    
    # lowest_num = find_lowest_integer(exp_folder)

    # copy_and_rename_files(exp_folder, lowest_num, exp_folder)

    # go over each net0-* file , and retrieve the min right part

    # copy paste the files with this number and remove the number part (on the copied versions)

    # $task $type $width_height $nb_examples CubeSpaceAE_AMA4Conv kltune2
    # run ./train_kltune.py dump and ./pddl-ama3.sh $dir ....

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
        width=""
        height=""
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

    args_list = [exp_folder]

    script_path = './pddl-ama3.sh'

    result = subprocess.run(['bash', script_path] + args_list, capture_output=False, text=True)


elif args.task == "transform_into_clustered_pddl":

    # check that two other args are defined


    if args.clustering_with_penalty is None:
        print("clustering_with_penalty arg required")
        exit()

    if args.clustering_base_data is None:
        print("clustering_base_data arg required")
        exit()

    if args.ors_in_whens_are_groups is None:
        print("ors_in_whens_are_groups arg required")
        exit()

    args_dict = {
        "--base_dir": exp_folder,
        "--data_folder": data_folder,
        "--clustering_base_data": args.clustering_base_data,
        "--clustering_with_penalty": bool_to_str(args.clustering_with_penalty),
        "--specific_whens": bool_to_str(args.specific_whens),
        "--ors_in_whens_are_groups": bool_to_str(args.ors_in_whens_are_groups)
    }
    args_list = []
    for key, value in args_dict.items():
        args_list.append(key)
        args_list.append(str(value))
  

    script_path = "construct_condpddl_Clustered_LowLvls.py"
    result = subprocess.run(['python', script_path] + args_list, capture_output=False, text=True)

elif args.task == "cluster_llas":

    # exp_folder

    args_dict = {
        "--base_dir": exp_folder,
        "--data_folder": data_folder
    }
    args_list = []
    for key, value in args_dict.items():
        args_list.append(key)
        args_list.append(str(value))


    script_path = "./cluster_llas.py"

    result = subprocess.run(['python', script_path] + args_list, capture_output=False, text=True)


# 

elif args.task == "gen_dfa_array":

    # exp_folder

    args_dict = {
        "--base_dir": exp_folder,
        "--data_folder": data_folder,
        "--exp_folder": exp_folder
    }
    args_list = []
    for key, value in args_dict.items():
        args_list.append(key)
        args_list.append(str(value))


    script_path = "./gen_dfa_array.py"

    result = subprocess.run(['python', script_path] + args_list, capture_output=False, text=True)



elif args.task == "gen_plans":

    ### from the domain PDDL generated 

    ## if in exp_folder there are several dir with the name pbs in it 

    ### THEN


    superfolders = [d for d in os.listdir(exp_folder) if "pbs" in d and os.path.isdir(os.path.join(exp_folder, d))]

    superfolders = ["pbs"]

    for superfold in superfolders:

        # print("SUPERFOLD IS ")
        # print(superfold)
        # exit()
        subfolders = [f.name for f in os.scandir(exp_folder+"/"+superfold) if f.is_dir()]

        print("subfoldersSSSSSSSSSSSSSS")
        print(subfolders)


        
        #domain_file = "domain.pddl"
        #domain_file = "domain_ORIGINAL.pddl"
        #domain_file = "domainCondBIS.pddl"
        #domain_file = "THENEWDOMAINBIS.pddl"
        #domain_file = "domainClustered_VeryHigh.pddl"
        #domain_file = "domainClustered_llas_True_only_effects_speWhens.pddl"
        domain_file = "domainClustered_llas_True_only_effects.pddl"

        if not "partial" in args.dataset_folder:
            

            for subfold in subfolders:
                

                heuri = "blind"
                # if "_" in superfold:
                #     heuri = superfold.split("_")[1]

                # 
                args_dict = {
                    exp_folder+"/"+domain_file: None,
                    exp_folder+"/"+superfold+"/"+subfold: None,
                    heuri: None, #"blind": None,
                    "1": None,
                    # "0.5": None,
                    # "spe_dom_dir ": "r_latplan_exps/"+args.domain+"/"+args.dataset_folder.split("/")[0]
                }
                # $dir/domain.pddl $dir/$probs_subdir blind 1


                # Convert the dictionary to a list of arguments
                args_list = []
                for key, value in args_dict.items():
                    if key == "width" or key == "height":
                        args_list.append(str(value))
                    else:
                        args_list.append(key)
                        if value is not None:
                            args_list.append(str(value))

                
                args_list.append(args.type)
                args_list.append("0.1")
                # if args.use_base_to_load:
                #     args_list.append("r_latplan_exps/"+args.domain+"/"+args.dataset_folder.split("/")[0])

                #result = subprocess.run(['python', script_path] + args_list, capture_output=False, text=True)
                #args_list.remove("spe_dom_dir")

                script_path = './ama3-planner.py'

       
 
                result = subprocess.run(['python', script_path] + args_list, capture_output=False, text=True)

                #exit()

        else:
            print("exp_folder")
            print(exp_folder)
            exp_folder = exp_folder + "/pbs/" + args.pb_folder


            args_dict = {
                exp_folder+"/"+domain_file: None,
                exp_folder: None,
                "blind": None,
                "1": None,

            }

            args_list = []
            for key, value in args_dict.items():
                if key == "width" or key == "height":
                    args_list.append(str(value))
                else:
                    args_list.append(key)
                    if value is not None:
                        args_list.append(str(value))

            args_list.append(args.type)
            #result = subprocess.run(['python', script_path] + args_list, capture_output=False, text=True)

            # print("args_list")
            # print(args_list)

            script_path = './ama3-planner.py'

            result = subprocess.run(['python', script_path] + args_list, capture_output=False, text=True)


    ### go through each Pb folder and use ama3-planner.py (to modify of course) on each





# in input

# directory of the exp

# 1) gen the pddl from the best weights


# 2) 