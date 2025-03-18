# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import math

# Get the angles from 0 to 2 pie (360 degree) in narray object


# Using built-in trigonometric function we can directly plot
# the given cosine wave for the given angles


# workspace/latplanRealOneHotActionsV2-fresh-hanoi-4-4-high-lvl-actions/samples/hanoi_4_4_5000_CubeSpaceAE_AMA4Conv/logs/rural-sweep-601/shap_vals_effects_no_touching





for ac_num in range(22):

    if ac_num == 5 or ac_num == 11:
        continue

    # Retrive the min / max among all

    max_vals = 0

    min_vals = 0

    nber_effects = 0


    #with open("shap_vals_persisting_effects_removed/action_"+str(ac_num)+"_ShapOfHighLvlEffect.txt") as file:
    with open("shap_vals_persisting_effects_removed/action_"+str(ac_num)+"_ShapOfHighLvlEffect_withEmptySet.txt") as file:

        all_vals = []

        for num_line, line in enumerate(file):

            line_stripped = line.strip()

            if "for effect" in line_stripped:
                title_subgraph = line_stripped.split(" ")[-1]
                nber_effects += 1

            elif "\n" == line:
                a= 1111
                print("for effect {}".format(str(title_subgraph)))
            else:
                all_vals.append(float(line_stripped))

    max_vals = max(all_vals)
    min_vals     = min(all_vals)

    print("max vals is {}".format(max_vals))

    print("min vals is {}".format(min_vals))
    X = np.arange(min_vals, max_vals, 0.5)
    if max_vals - min_vals < 0.5:
        X = [-1, 0, 1]

    # un carré, sqrt(nbre effets)

    # 3+1

    # int(sqrt(nber_effets+1))    
    #
    #       is_integer()
    # Initialise the subplot function using number of rows and columns
    sqrt = math.sqrt(nber_effects)
    size_side = 0
    if not sqrt.is_integer():
        size_side = int(sqrt)+1
    else:
        size_side = int(sqrt)

    
    print("NBER EFFECTS {}".format(nber_effects))
    print("size side is {}".format(size_side))
    if size_side == 1:
        size_side = 2

    figure, axis = plt.subplots(size_side, size_side)
    figure.tight_layout()



    #with open("shap_vals_persisting_effects_removed/action_"+str(ac_num)+"_ShapOfHighLvlEffect.txt") as file:
    with open("shap_vals_persisting_effects_removed/action_"+str(ac_num)+"_ShapOfHighLvlEffect_withEmptySet.txt") as file:

        title_subgraph = ""
        tmp_vals = []
        num_plot = 0

        for num_line, line in enumerate(file):

            line_stripped = line.strip()

            if "for effect" in line_stripped:
                title_subgraph = line_stripped.split(" ")[-1]
                tmp_vals = []

            elif "\n" == line:
                print("for effect {}".format(str(title_subgraph)))
                print(tmp_vals)
                #print(num_line)

                ii = num_plot % size_side
                jj = num_plot // size_side

                # 
                #axis[ii, jj].plot(X, tmp_vals)
                # if ac_num == 17:
                #     print("XX is {}".format(X))
                #     exit()
                axis[ii, jj].hist(tmp_vals, bins=X)
                axis[ii, jj].set_title(title_subgraph)

                num_plot += 1

            else:
                tmp_vals.append(float(line_stripped))

    plt.savefig("histogramAction_"+str(ac_num)+"_persist_effects_removed_ShapsOfHighLvlEffects_WithEmptySet.png")    


# # For Cosine Function
# axis[0, 1].plot(X, Y2)
# axis[0, 1].set_title("Cosine Function")

# # For Tangent Function
# axis[1, 0].plot(X, Y3)
# axis[1, 0].set_title("Tangent Function")

# # For Tanh Function
# axis[1, 1].plot(X, Y4)
# axis[1, 1].set_title("Tanh Function")

# Combine all the operations and display
