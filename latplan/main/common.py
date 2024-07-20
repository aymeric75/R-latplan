import os
import os.path
import glob

import numpy as np

import latplan.util.stacktrace
from latplan.util.tuning import simple_genetic_search, parameters, nn_task, reproduce, load_history
from latplan.util        import curry
import sys
import json
import random


################################################################
# globals

args     = None
sae_path = None

################################################################
# command line parsing

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)


# 
# RawDescriptionHelpFormatter
# ArgumentDefaultsHelpFormatter


parser.add_argument(
    "mode",
    help=(
        "A string which contains mode substrings."
        "\nRecognized modes are:"
        "\n" 
        "\n   learn     : perform the training with a hyperparameter tuner. Results are stored in samples/[experiment]/logs/[hyperparameter]."
        "\n               If 'learn' is not specified, it attempts to load the stored weights."
        "\n   plot      : produce visualizations"
        "\n   dump      : dump the csv files necessary for producing the PDDL models"
        "\n   summary   : perform extensive performance evaluations and collect the statistics, store the result in performance.json"
        "\n   debug     : debug training limited to epoch=2, batch_size=100. dataset is truncated to 200 samples"
        "\n   reproduce : train the best hyperparameter so far three times with different random seeds. store the best results."
        "\n   iterate   : iterate plot/dump/summary commands above over all hyperparmeters that are already trained and stored in logs/ directory."
        "\n"
        "\nFor example, learn_plot_dump contains 'learn', 'plot', 'dump' mode."
        "\nThe separater does not matter because its presense is tested by python's `in` directive, i.e., `if 'learn' in mode:` ."
        "\nTherefore, learnplotdump also works."))


subparsers = parser.add_subparsers(
    title="subcommand",
    metavar="subcommand",
    required=True,
    description=(
        "\nA string which matches the name of one of the dataset functions in latplan.main module."
        "\n"
        "\nEach task has a different set of parameters, e.g.,"
        "\n'puzzle' has 'type', 'width', 'height' where 'type' should be one of 'mnist', 'spider', 'mandrill', 'lenna',"
        "\nwhile 'lightsout' has 'type' being either 'digital' and 'twisted', and 'size' being an integer."
        "\nSee subcommand help."))

def add_common_arguments(subparser,task,objs=False):
    subparser.set_defaults(task=task)
    subparser.add_argument(
        "num_examples",
        default=5000,
        type=int,
        help=(
            "\nNumber of data points to use. 90%% of this number is used for training, and 5%% each for validation and testing."
            "\nIt is assumed that the user has already generated a dataset archive in latplan/puzzles/,"
            "\nwhich contains a larger number of data points using the setup-dataset script provided in the root of the repository."))
    subparser.add_argument(
        "aeclass",
        help=
        "A string which matches the name of the model class available in latplan.model module.\n"+
        "It must be one of:\n"+
        "\n".join([ " "*4+name for name, cls in vars(latplan.model).items()
                    if type(cls) is type and \
                    issubclass(cls, latplan.network.Network) and \
                    cls is not latplan.network.Network
                ])
    )
    if objs:
        subparser.add_argument("location_representation",
                               nargs='?',
                               choices=["bbox","coord","binary","sinusoidal","anchor"],
                               default="coord",
                               help="A string which specifies how to convert/encode the location in the dataset. See documentations for normalize_transitions_objects")
        subparser.add_argument("randomize_location",
                               nargs='?',
                               type=bool,
                               default=False,
                               help="A boolean which specifies whether we randomly translate the environment globally. See documentations for normalize_transitions_objects")
    subparser.add_argument("comment",
                           nargs='?',
                           default="",
                           help="A string which is appended to the directory name to label each experiment.")


    subparser.add_argument("--dataset_folder",
                           nargs='?',
                           default="",
                           help="folder path of where the images are")

    subparser.add_argument("--type",
                           nargs='?',
                           default="",
                           help="if vanilla or r_latplan")
    
    return




def load_dataset(path_to_file):
    import pickle
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data


def main(parameters={}):
    import latplan.util.tuning
    latplan.util.tuning.parameters.update(parameters)

    import sys
    global args, sae_path
    args = parser.parse_args()
    task = args.task

    delattr(args,"task")
    latplan.util.tuning.parameters.update(vars(args))

    print("'sys.argvsys.argvsys.argv")
    print(sys.argv[2])

    print(sys.argv)
    
    if sys.argv[-1] == 'vanilla':
        import latplan.modelVanilla
    else:
        import latplan.model


    if(sys.argv[2]=="puzzle"):
        sae_path = "_".join(sys.argv[2:9])
    if(sys.argv[2]=="blocks"):
        sae_path = "_".join(sys.argv[2:7])
    if(sys.argv[2]=="lightsout"):
        sae_path = "_".join(sys.argv[2:8])
    if(sys.argv[2]=="sokoban"):
        sae_path = "_".join(sys.argv[2:7])
    if(sys.argv[2]=="hanoi"):
        sae_path = "_".join(sys.argv[2:7])


    
    print(vars(args))
    # latplan.util.tuning.parameters.update(vars(args))
    # sae_path = "_".join(sys.argv[2:])

    print("SAE PATH")
    print(sae_path)


    try:
        task(args)
    except:
        latplan.util.stacktrace.format()


################################################################
# procedures for each mode

def plot_autoencoding_image(ae,transitions,label):
    if 'plot' not in args.mode:
        return

    if hasattr(ae, "plot_transitions"):
        transitions = transitions[:6]
        ae.plot_transitions(transitions, ae.local(f"transitions_{label}"),verbose=True)
    else:
        transitions = transitions[:3]
        states = transitions.reshape((-1,*transitions.shape[2:]))
        ae.plot(states, ae.local(f"states_{label}"),verbose=True)

    return


def dump_all_actions(ae,configs,trans_fn,name = "all_actions.csv",repeat=1):
    if 'dump' not in args.mode:
        return
    l     = len(configs)
    batch = 5000
    loop  = (l // batch) + 1
    print(ae.local(name))
    with open(ae.local(name), 'wb') as f:
        for i in range(repeat):
            for begin in range(0,loop*batch,batch):
                end = begin + batch
                print((begin,end,len(configs)))
                transitions = trans_fn(configs[begin:end])
                pre, suc    = transitions[0], transitions[1]
                pre_b       = ae.encode(pre,batch_size=1000).round().astype(int)
                suc_b       = ae.encode(suc,batch_size=1000).round().astype(int)
                actions     = np.concatenate((pre_b,suc_b), axis=1)
                np.savetxt(f,actions,"%d")


def dump_actions(ae,transitions,name = "actions.csv",repeat=1):
    if 'dump' not in args.mode:
        return
    print(ae.local(name))
    ae.dump_actions(transitions,batch_size = 1000)


def dump_all_states(ae,configs,states_fn,name = "all_states.csv",repeat=1):
    if 'dump' not in args.mode:
        return
    l     = len(configs)
    batch = 5000
    loop  = (l // batch) + 1
    print(ae.local(name))
    with open(ae.local(name), 'wb') as f:
        for i in range(repeat):
            for begin in range(0,loop*batch,batch):
                end = begin + batch
                print((begin,end,len(configs)))
                states   = states_fn(configs[begin:end])
                states_b = ae.encode(states,batch_size=1000).round().astype(int)
                np.savetxt(f,states_b,"%d")


def dump_states(ae,states,name = "states.csv",repeat=1):
    if 'dump' not in args.mode:
        return
    print(ae.local(name))
    with open(ae.local(name), 'wb') as f:
        for i in range(repeat):
            np.savetxt(f,ae.encode(states,batch_size = 1000).round().astype(int),"%d")


def dump_code_unused():
    # This code is not used. Left here for copy-pasting in the future.
    if False:
        dump_states      (ae,all_states,"all_states.csv")
        dump_all_actions (ae,all_transitions_idx,
                          lambda idx: all_states[idx.flatten()].reshape((len(idx),2,num_objs,-1)).transpose((1,0,2,3)))


def train_val_test_split(x):
    train = x[:int(len(x)*0.9)]
    val   = x[int(len(x)*0.9):int(len(x)*0.95)]
    test  = x[int(len(x)*0.95):]
    return train, val, test


def run(path,transitions,extra=None):


    #### former Vanilla Latplan code
    train, val, test = train_val_test_split(transitions)

    # learning: un base_aux_json
    #                   puis le exp_aux_json
    #
    #
    #       testing depuis le exp_aux_json

    # 
    dataset_fold = None

    if args.type == "vanilla":
        dataset_fold = "r_vanilla_latplan_datasets"
    else:
        dataset_fold = "r_latplan_datasets"


    dataset_aux_json_folder_base = dataset_fold+"/"+sys.argv[2]

    dataset_aux_json_folder_exp = dataset_fold+"/"+sys.argv[2] + "/" + args.dataset_folder

    exp_aux_json_folder = None
    if args.type == "vanilla":
        exp_aux_json_folder = "r_vanilla_latplan_exps/" +sys.argv[2] + "/" + args.dataset_folder
    else:
        exp_aux_json_folder = "r_latplan_exps/" +sys.argv[2] + "/" + args.dataset_folder
    
    # 
    path_to_dataset = dataset_aux_json_folder_exp +"/data.p"


    # load dataset for the specific experiment
    loaded_data = load_dataset(path_to_dataset)
    
    train_set = loaded_data["train_set"] 
    test_val_set = loaded_data["test_val_set"] 
    all_pairs_of_images_reduced_orig = loaded_data["all_pairs_of_images_reduced_orig"] 
    all_actions_one_hot = loaded_data["all_actions_one_hot"]
    mean_all = loaded_data["mean_all"] 
    std_all = loaded_data["std_all"] 
    all_actions_unique = loaded_data["all_actions_unique"] 
    orig_max = loaded_data["orig_max"] 
    orig_min = loaded_data["orig_min"] 


    if args.type == "vanilla":
        train_set_ = []
        for tr in train_set:
            train_set_.append(tr[0])
        train_set = np.array(train_set_)
        test_val_set_ = []
        for tr in test_val_set:
            test_val_set_.append(tr[0])
        test_val_set = np.array(test_val_set_)



    # load the json file from the base domain folder (in order to update and copy/save it in the exp subfolder)
    
    if 'learn' in args.mode:
        print("LI1")
        print(os.path.join("r_latplan_exps/" +sys.argv[2],"aux.json"))

        if os.path.isfile(os.path.join("r_latplan_exps/" +sys.argv[2],"aux.json")):
            with open(os.path.join("r_latplan_exps/" +sys.argv[2],"aux.json"),"r") as f:
                data = json.load(f)

    elif 'dump' in args.mode:
        print("LI2")
        print(os.path.join(exp_aux_json_folder,"aux.json"))
        if os.path.isfile(os.path.join(exp_aux_json_folder,"aux.json")):
            with open(os.path.join(exp_aux_json_folder,"aux.json"),"r") as f:
                data = json.load(f)


    # Step 2: Replace 'mean' and 'std' in the dictionary
    data['parameters']['mean'] = mean_all.tolist()
    data['parameters']['std'] = std_all.tolist()

    data['parameters']['orig_max'] = orig_max
    data['parameters']['orig_min'] = orig_min
    data["parameters"]["time_start"] = ""

    if args.type == "vanilla" and 'dump' in args.mode:
        data["parameters"]["beta_z_and_beta_d"] = [1, 1000]
        data["parameters"]["pdiff_z1z2_z0z3"] = [1, 1000]



    if args.type == "vanilla":
        data['parameters']['A'] = 6000
    else:
        data['parameters']['A'] = len(all_actions_unique)

    if args.type == "vanilla":

        aaa = [2]
        aaa.extend(train_set[0][0].shape)
        data["input_shape"] = aaa

    else:

        aaa = [2]
        aaa.extend(train_set[0][0][0].shape)
        data["input_shape"] = aaa


    print("ON EST LAAAAAAAAAA")
    print("dataset_aux_json_folder_exp {}".format(dataset_aux_json_folder_exp))
    print()
    print("exp_aux_json_folder {}".format(exp_aux_json_folder))
    print()
    # save the updated aux.json into the exp subfolder of the dataset folder
    with open(os.path.join(dataset_aux_json_folder_exp,"aux.json"),"w") as f:
        json.dump(data, f, indent=4)

    # save the updated aux.json into the exp folder (in r_latplan_exps)
    with open(os.path.join(exp_aux_json_folder,"aux.json"),"w") as f:
        json.dump(data, f, indent=4)
    

    # finally, read the saved exp aux.json (see above)
    with open(os.path.join(dataset_aux_json_folder_exp,"aux.json"),"r") as f:
        parameters = json.load(f)["parameters"]


    parameters["mean"] = mean_all
    parameters["std"] = std_all

    val_set = test_val_set[:len(test_val_set)//2]

    random.shuffle(val_set)



    def postprocess(ae):
        show_summary(ae, train, test)
        plot_autoencoding_image(ae,train,"train")
        plot_autoencoding_image(ae,test,"test")
        dump_actions(ae,transitions)
        return ae


    def report(net,eval):
        try:
            postprocess(net)
            if extra:
                extra(net)
        except:
            latplan.util.stacktrace.format()
        return



    if 'dump' in args.mode:

        # prob ici c'est que ça load 

        parameters["time_start"] = ""

        # beta_z_and_beta_d
        parameters["epoch"] = 1
        #parameters["A"] = 6000

        parameters["beta_ama_recons"] = 1
        parameters["beta_z_and_beta_d"] = [1, 1000]
        parameters["pdiff_z1z2_z0z3"] = [1, 1000]
        print("theparameters")
        #print(parameters)

        # print("exp_aux_json_folderexp_aux_json_folder")
        # print(exp_aux_json_folder)
        # exit()
        if sys.argv[-1] == 'vanilla':
            net = latplan.modelVanilla.load(exp_aux_json_folder, allow_failure=False)
        else:
            net = latplan.model.load(exp_aux_json_folder, allow_failure=False)

        print(type(transitions)) # 
        #print(transitions.shape) # (5000, 2, 48, 48, 1)
        print(len(transitions))
        
        # print(np.array(transitions).shape)

        if args.type == "vanilla":
            #dump_actions(net, [transitions, actions_transitions], name = "actions.csv", repeat=1)
            dump_actions(net, train_set, name = "actions.csv", repeat=1)
        
        else:

            thetransarray=[]
            theactionarray=[]
            for trr in train_set:
                thetransarray.append(trr[0])
                theactionarray.append(trr[1])
            

            #dump_actions(net, [transitions, actions_transitions], name = "actions.csv", repeat=1)
            dump_actions(net, [thetransarray, theactionarray], name = "actions.csv", repeat=1)
            # alors...






    if 'learn' in args.mode:


        parameters["epoch"] = 10000

        parameters["load_sae_weights"] = False
        
        parameters["use_wandb"] = True


        # train_set = [train_set[0]]
        # val_set = [val_set[0]]

        parameters["the_exp_path"] = exp_aux_json_folder
        # parameters["beta_z_and_beta_d"] = [10, 1000]
        # parameters["N"] = 300
        # parameters["pdiff_z1z2_z0z3"] = 0
        parameters["type"] = args.type
        
        if args.type == "vanilla":
            task = curry(nn_task, latplan.modelVanilla.get(parameters["aeclass"]), exp_aux_json_folder, train_set, train_set, val_set, val_set, parameters, False) 
            task()
        else:
            task = curry(nn_task, latplan.model.get(parameters["aeclass"]), exp_aux_json_folder, train_set, train_set, val_set, val_set, parameters, False) 
            task()

        # simple_genetic_search(
        #     curry(nn_task, latplan.model.get(parameters["aeclass"]),
        #           path,
        #           train, train, val, val), # noise data is used for tuning metric
        #     parameters,
        #     path,
        #     limit              = 100,
        #     initial_population = 100,
        #     population         = 100,
        #     report             = report,
        # )

    if 'resume' in args.mode:
        simple_genetic_search(
            lambda parameters: nn_task(latplan.model.get(parameters["aeclass"]), path, train, train, val, val, parameters, resume=True),
            parameters,
            path,
            limit              = 100,
            initial_population = 100,
            population         = 100,
            report             = report,
        )

    if 'debug' in args.mode:
        print("debug run. removing past logs...")
        for _path in glob.glob(os.path.join(path,"*")):
            if os.path.isfile(_path):
                os.remove(_path)
        parameters["epoch"]=1
        parameters["batch_size"]=100
        train, val = train[:200], val[:200]
        simple_genetic_search(
            curry(nn_task, latplan.model.get(parameters["aeclass"]),
                  path,
                  train, train, val, val), # noise data is used for tuning metric
            parameters,
            path,
            limit              = 1,
            initial_population = 1,
            population         = 1,
            report             = report,
        )

    if 'reproduce' in args.mode:   # reproduce the best result from the grid search log
        reproduce(
            curry(nn_task, latplan.model.get(parameters["aeclass"]),
                  path,
                  train, train, val, val), # noise data is used for tuning metric
            path,
            report      = report,
        )

    if 'iterate' in args.mode:
        open_list, _ = load_history(path)
        topk = open_list[:10]
        topk_dirnames = [
            os.path.join(path,"logs",elem[1]["hash"])
            for elem in topk
        ]
        print(f"iterating mode {args.mode} for all weights stored under logs")
        for path in topk_dirnames:
            postprocess(latplan.model.load(path))



def show_summary(ae,train,test):
    if 'summary' in args.mode:
        ae.summary()
        ae.report(train, test_data = test, train_data_to=train, test_data_to=test)


