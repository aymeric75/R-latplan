#!/usr/bin/env python3


no_inv = " --translate-options --invariant-generation-max-candidates 0 "

limit_time = " --search-time-limit 60s " + no_inv


options = {
    "lmcut" : "--search astar(lmcut())",
    "blind" : "--search astar(blind())",
    "hmax"  : "--search astar(hmax())",
    "ff"    : "--search eager(single(ff()))",
    "lff"   : "--search lazy_greedy(ff())",
    "lffpo" : "--evaluator h=ff() --search lazy_greedy(h, preferred=h)",
    "gc"    : "--search eager(single(goalcount()))",
    "lgc"   : "--search lazy_greedy(goalcount())",
    "lgcpo" : "--evaluator h=goalcount() --search lazy_greedy(h, preferred=h)",
    "cg"    : "--search eager(single(cg()))",
    "lcg"   : "--search lazy_greedy(cg())",
    "lcgpo" : "--evaluator h=cg() --search lazy_greedy(h, preferred=h)",
    "lama"  : "--alias lama-first",
    "oldmands" : "--search astar(merge_and_shrink(shrink_strategy=shrink_bisimulation(max_states=50000,greedy=false),merge_strategy=merge_dfp(),label_reduction=exact(before_shrinking=true,before_merging=false)))",
    "mands"    : "--search astar(merge_and_shrink(shrink_strategy=shrink_bisimulation(greedy=false),merge_strategy=merge_sccs(order_of_sccs=topological,merge_selector=score_based_filtering(scoring_functions=[goal_relevance,dfp,total_order])),label_reduction=exact(before_shrinking=true,before_merging=false),max_states=50k,threshold_before_merge=1))",
    "pdb"   : "--search astar(pdb())",
    "cpdb"  : "--search astar(cpdbs())",
    "ipdb"  : "--search astar(ipdb())",
    "zopdb"  : "--search astar(zopdbs())",
}



##################  CAREFUL !!!!!!!!!!!!!!!!  THIS OFTEN PREVENTs PLANS TO BE GENERATED !!!!!!!!!!!!!!!
# for k, v in options.items():
#     options[k] = v + limit_time


import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("domainfile", help="pathname to a PDDL domain file")
parser.add_argument("problem_dir", help="pathname to a directory containing init.png and goal.png")
parser.add_argument("heuristics", choices=options.keys(),
                    help="heuristics configuration passed to fast downward. The details are:\n"+
                    "\n".join([ " "*4+key+"\n"+" "*8+value for key,value in options.items()]))
parser.add_argument("cycle", type=int, default=1, nargs="?",
                    help="number of autoencoding cycles to perform on the initial/goal images")
parser.add_argument("typee", type=str, default=None, nargs="?",
                    help="vanilla or r latplan")
parser.add_argument("sigma", type=float, default=None, nargs="?",
                    help="sigma of the Gaussian noise added to the normalized initial/goal images.")

parser.add_argument("network_dir_spe", type=str, default=None, nargs="?",
                    help="Optional: Path to the specific network directory.")



# ['r_latplan_exps/hanoi/hanoi_complete_clean_faultless_withoutTI/confs/0__true__true__true__x_all_atoms_dt2/domainCondBIS.pddl', 
# 'r_latplan_exps/hanoi/hanoi_complete_clean_faultless_withoutTI/confs/0__true__true__true__x_all_atoms_dt2/pbs_lmcut/1_0', 
# 'lmcut', 
# '1', 
# 'spe_dom_dir ', 
# 'r_latplan_exps/hanoi/hanoi_complete_clean_faultless_withoutTI', 
# 'r_latplan']

args = parser.parse_args()


import subprocess
import os
import sys
import latplan
import latplan.model
from latplan.util import *
from latplan.util.planner import *
from latplan.util.plot import *
import latplan.util.stacktrace
import os.path
import keras.backend as K
import tensorflow as tf
import math
import time
import json

import numpy as np
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(threshold=sys.maxsize,formatter={'float_kind':float_formatter})


def main(domainfile, problem_dir, heuristics, cycle, typee, sigma=None, network_dir_spe=None):



    sigma = None
    network_dir = os.path.dirname(domainfile)
    domainfile_rel = os.path.relpath(domainfile, network_dir)

    if args.network_dir_spe is not None:
        network_dir = args.network_dir_spe

    print("begiin")

    print(network_dir) # r_latplan_exps/hanoi/hanoi_complete_clean_faultless
    print(domainfile_rel) # domain.pddl

    def domain(path):
        dom_prefix = domainfile_rel.replace("/","_")
        root, ext = os.path.splitext(path)
        return "{}_{}{}".format(os.path.splitext(dom_prefix)[0], root, ext)
    def heur(path):
        root, ext = os.path.splitext(path)
        return "{}_{}{}".format(heuristics, root, ext)
    
    log("loaded puzzle")
    if typee == "vanilla":
        sae = latplan.modelVanilla.load(network_dir,allow_failure=True)
    else:
        sae = latplan.model.load(network_dir,allow_failure=True)
    log("loaded sae")
    setup_planner_utils(sae, problem_dir, network_dir, "ama3")


    p = puzzle_module(sae)
    log("loaded puzzle")

    log(f"loading init/goal")
    if "sokoban" in network_dir:
        init, goal = init_goal_misc(p,cycle,noise=sigma, image_path=problem_dir, is_soko=True)
    else:
        init, goal = init_goal_misc(p,cycle,noise=sigma, image_path=problem_dir, is_soko=False)
    log(f"loaded init/goal")

    log(f"start planning")

    bits = np.concatenate((init,goal)) 

    ###### files ################################################################
    ig          = problem(ama(network(domain(heur(f"problem.ig")))))
    problemfile = problem(ama(network(domain(heur(f"problem.pddl")))))
    planfile    = problem(ama(network(domain(heur(f"problem.plan")))))
    tracefile   = problem(ama(network(domain(heur(f"problem.trace")))))
    csvfile     = problem(ama(network(domain(heur(f"problem.csv")))))
    pngfile     = problem(ama(network(domain(heur(f"problem.png")))))
    jsonfile    = problem(ama(network(domain(heur(f"problem.json")))))
    logfile     = problem(ama(network(domain(heur(f"problem.log")))))
    npzfile     = problem(ama(network(domain(heur(f"problem.npz")))))
    negfile     = problem(ama(network(domain(heur(f"problem.negative")))))

    


    valid = False
    found = False
    try:
        ##### preprocessing ################################################################
        log(f"start generating problem")
        os.path.exists(ig) or np.savetxt(ig,[bits],"%d")
        echodo(["helper/ama3-problem.sh",ig,problemfile])
        log(f"finished generating problem")


        # CODE FOR REMOVING THE NEGATIVE STATES IN THE INIT STATE (AND REMOVING NON PRESENT "EFFECTS" in the goal state)
        # Save the current path
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the new path (relative to the current file)
        translate_path = os.path.join(current_dir, "downward", "src", "translate")

        # Temporarily add it to sys.path
        sys.path.insert(0, translate_path)

        try:
            #import options  # Replace with actual module name
            import FDgrounder
            from FDgrounder import pddl_parser as pddl_pars


            task = pddl_pars.open(
                domain_filename=domainfile, task_filename=problemfile) # options.task
                




            # ############ CODE FOR REMOVING NON PRESENT "EFFECTS" in the goal states ##################
            # effectsToKeep = []
            # for trans_id, act in enumerate(task.actions):
            #     tmp_act_effects = act.effects

            #     for eff in tmp_act_effects:
            #         # print("EFF LITERAL")
            #         # print(eff.literal)
            #         integer = int(''.join(x for x in str(eff.literal) if x.isdigit()))
            #         transformed_name = ""
            #         if "Negated" in str(eff.literal):
            #             transformed_name += "del_"+str(integer)
            #         else:
            #             transformed_name += "add_"+str(integer)

            #         if transformed_name not in effectsToKeep:
            #             effectsToKeep.append(transformed_name)

            # goal_parts = list(task.goal.parts)

            # for part in goal_parts:
            #     integer = int(''.join(x for x in str(part) if x.isdigit()))
            #     transformed_name = ""
            #     if "Negated" in str(part):
            #         transformed_name += "del_"+str(integer)
            #     else:
            #         transformed_name += "add_"+str(integer)

            #     print("transformed_name is {}".format(transformed_name))

            #     if transformed_name not in effectsToKeep:
            #         goal_parts.remove(part)
            # task.goal.parts = tuple(goal_parts)


            # ############ END CODE FOR REMOVING NON PRESENT "EFFECTS" in the goal states ##################




            f = open(problemfile, "w")
            task.domain_name = "latent"
            f.write(task.get_pddl_problem())
            f.close()

        finally:
            # Restore sys.path to avoid conflicts
            sys.path.pop(0)




























        print("R LATPLAN STUFF")
        # print(problemfile)
        # problemfile = "r_latplan_exps/hanoi/hanoi_complete_clean_faultless_withoutTI/pbs/8_0/ama3_r_latplan_exps_hanoi_hanoi_complete_clean_faultless_withoutTI_domain_blind_problem.pddl"
        print("problemfile")
        print(problemfile)
        print(options[heuristics])

        ###### do planning #############################################
        log(f"start planning")
        echodo(["helper/fd-latest.sh", options[heuristics], problemfile, domainfile])
        log(f"finished planning")
        print("FINITO")

        

        # if not os.path.exists(planfile):
        #     return valid
        found = True
        log(f"start running a validator")
        #echodo(["arrival", "-r", domainfile, problemfile, planfile, tracefile])
        echodo(["arrival", domainfile, problemfile, planfile, tracefile])
        log(f"finished running a validator")

        log(f"start parsing the plan")
        with open(csvfile,"w") as f:
            echodo(["lisp/ama3-read-latent-state-traces.bin", tracefile, str(len(init))],
                   stdout=f)
        plan = np.loadtxt(csvfile, dtype=int)
        log(f"finished parsing the plan")

        if plan.ndim != 2:
            assert plan.ndim == 1
            print("Found a plan with length 0; single state in the plan.")
            return valid

        log(f"start plotting the plan")
        sae.plot_plan(plan, pngfile, verbose=True)
        log(f"finished plotting the plan")

        log(f"start archiving the plan")
        plan_images = sae.decode(plan)
        np.savez_compressed(npzfile,img_states=plan_images)
        log(f"finished archiving the plan")

        log(f"start visually validating the plan image : transitions")
        # note: only puzzle, hanoi, lightsout have the custom validator, which are all monochrome.
        plan_images = sae.render(plan_images) # unnormalize the image

        return True

    finally:
        with open(jsonfile,"w") as f:
            parameters = sae.parameters.copy()
            del parameters["mean"]
            del parameters["std"]
            json.dump({
                "network":network_dir,
                "problem":os.path.normpath(problem_dir).split("/")[-1],
                "domain" :os.path.normpath(problem_dir).split("/")[-2],
                "noise":sigma,
                "times":times,
                "heuristics":heuristics,
                "domainfile":domainfile,
                "problemfile":problemfile,
                "planfile":planfile,
                "tracefile":tracefile,
                "csvfile":csvfile,
                "pngfile":pngfile,
                "jsonfile":jsonfile,
                "statistics": "", #json.loads(echo_out(["helper/fd-parser.awk", logfile])),
                "parameters":parameters,
                "valid":valid,
                "found":found,
                "exhausted": os.path.exists(negfile),
                "cycle":cycle,
            }, f, indent=2)



if __name__ == '__main__':
    try:

        main(**vars(args))
    except:
        import latplan.util.stacktrace
        latplan.util.stacktrace.format()
