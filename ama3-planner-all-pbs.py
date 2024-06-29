#!/usr/bin/env python3

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


import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("domainfile", help="pathname to a PDDL domain file")
parser.add_argument("problem_dir", help="pathname to a directory containing init.png and goal.png")
parser.add_argument("heuristics", choices=options.keys(),
                    help="heuristics configuration passed to fast downward. The details are:\n"+
                    "\n".join([ " "*4+key+"\n"+" "*8+value for key,value in options.items()]))
parser.add_argument("cycle", type=int, default=1, nargs="?",
                    help="number of autoencoding cycles to perform on the initial/goal images")
parser.add_argument("sigma", type=str, default=None, nargs="?",
                    help="sigma of the Gaussian noise added to the normalized initial/goal images.")

parser.add_argument("extension", type=str, default=None, nargs="?",
                    help="extension for name of image files.")
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


def main(domainfile, problem_dir, heuristics, cycle, sigma, extension=""):


    sigma = None

    
    network_dir = os.path.dirname(domainfile)
    domainfile_rel = os.path.relpath(domainfile, network_dir)
    
    def domain(path):
        dom_prefix = domainfile_rel.replace("/","_")
        root, ext = os.path.splitext(path)
        return "{}_{}{}".format(os.path.splitext(dom_prefix)[0], root, ext)
    def heur(path):
        root, ext = os.path.splitext(path)
        return "{}_{}{}".format(heuristics, root, ext)
    


    # Convert to grayscale using the luminosity method
    def rgb_to_grayscale(rgb_images):
        grayscale = np.dot(rgb_images[...,:3], [0.21, 0.72, 0.07])
        return np.stack((grayscale,)*3, axis=-1)


    print("in ama3-planner 1")
    print(network_dir)
    log("loaded puzzle")

    
    sae = latplan.model.load(network_dir, allow_failure=True)

    print(sae.parameters)


    log("loaded sae")
    print(sae)
    print("problem_dir")
    print(problem_dir)

    for root, dirs, files in os.walk(problem_dir):
        print("washere1")
        if dirs:
            for cc, d in enumerate(dirs):


                sub_problem_dir = problem_dir+"/"+d


                setup_planner_utils(sae, sub_problem_dir, network_dir, "ama3")
                print("in ama3-planner 2")
                
                p = puzzle_module(sae)
                log("loaded puzzle 2")

                log(f"loading init/goal")
                # init and goal must be as the latent space repr
                # meaning, for hanoi pddlgym, the init.png and goal.png images
                # (which were already 'reduced') must now
                # 1) be preprocessed
                # 2) be normalized
                # 3) passed through the encoder


                directory = '/workspace/latplanRealOneHotActionsV2-fresh-hanoi-4-4/all_states'
                counterr = 0
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)

                    

                    print("filepath: {}".format(filepath))

                    init, goal = init_goal_misc(p,cycle,noise=sigma, init_file_path = filepath)
                    log(f"loaded init/goal")
                    # init = rgb_to_grayscale(init)
                    # goal = rgb_to_grayscale(goal)

                    log(f"start planning")

                    print("THEINIT")
                    print(init)


                    bits = np.concatenate((init,goal))

                    ###### files ################################################################
                    ig          = problem(ama(network(domain(heur(str(counterr)+f"problem.ig")))))
                    problemfile = problem(ama(network(domain(heur(str(counterr)+f"problem.pddl")))))
                    planfile    = problem(ama(network(domain(heur(f"problem.plan")))))
                    tracefile   = problem(ama(network(domain(heur(f"problem.trace")))))
                    csvfile     = problem(ama(network(domain(heur(f"problem.csv")))))
                    pngfile     = problem(ama(network(domain(heur(f"problem.png")))))
                    jsonfile    = problem(ama(network(domain(heur(f"problem.json")))))
                    logfile     = problem(ama(network(domain(heur(f"problem.log")))))
                    npzfile     = problem(ama(network(domain(heur(f"problem.npz")))))
                    negfile     = problem(ama(network(domain(heur(f"problem.negative")))))

                    counterr += 1

                    valid = False
                    found = False


                    try:
                        ###### preprocessing ################################################################
                        log(f"start generating problem")
                        os.path.exists(ig) or np.savetxt(ig,[bits],"%d")
                        echodo(["helper/ama3-problem.sh", ig, problemfile])
                        log(f"finished generating problem")
                        continue
                        ###### do planning #############################################
                        log(f"start planning")
                        echodo(["helper/fd-latest.sh", options[heuristics], problemfile, domainfile])
                        log(f"finished planning")
                        # if not os.path.exists(planfile):
                        #     return valid
                        found = True
                        log(f"start running a validator")
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
                            #return valid
                            continue

                        print("planplan")
                        print(plan)

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
                        print('plan omages')
                        print(plan_images.shape)
                        continue

                        validation = p.validate_transitions([plan_images[0:-1], plan_images[1:]])
                        print(validation)
                        valid = bool(np.all(validation))
                        log(f"finished visually validating the plan image : transitions")

                        log(f"start visually validating the plan image : states")
                        print(p.validate_states(plan_images))
                        log(f"finished visually validating the plan image : states")
                        valid = True
                        #return valid

                    finally:
                        print("ok")
                        # with open(jsonfile,"w") as f:
                        #     parameters = sae.parameters.copy()
                        #     del parameters["mean"]
                        #     del parameters["std"]
                        #     json.dump({
                        #         "network":network_dir,
                        #         "problem":os.path.normpath(sub_problem_dir).split("/")[-1],
                        #         "domain" :os.path.normpath(sub_problem_dir).split("/")[-2],
                        #         "noise":sigma,
                        #         "times":times,
                        #         "heuristics":heuristics,
                        #         "domainfile":domainfile,
                        #         "problemfile":problemfile,
                        #         "planfile":planfile,
                        #         "tracefile":tracefile,
                        #         "csvfile":csvfile,
                        #         "pngfile":pngfile,
                        #         "jsonfile":jsonfile,
                        #         "statistics":json.loads(echo_out(["helper/fd-parser.awk", logfile])),
                        #         "parameters":parameters,
                        #         "valid":valid,
                        #         "found":found,
                        #         "exhausted": os.path.exists(negfile),
                        #         "cycle":cycle,
                        #     }, f, indent=2)
                    


if __name__ == '__main__':
    
    try:
        main(**vars(args))
    except:
        import latplan.util.stacktrace
        latplan.util.stacktrace.format()
