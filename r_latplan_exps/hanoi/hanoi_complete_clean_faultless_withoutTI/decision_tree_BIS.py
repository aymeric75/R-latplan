import pandas as pd
import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz, plot_tree
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os 
import pickle
import inspect
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
import joblib
import math
from multiprocessing import Pool, cpu_count, Manager, Value, Lock
from functools import partial

from skmultilearn.problem_transform import ClassifierChain
from lightgbm import LGBMClassifier
import lightgbm as lgb
from tqdm import tqdm

import h5py

from itertools import chain


# from lightgbm import LGBMClassifier
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split

# Generate synthetic dataset
#X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)




# with h5py.File('X_train.h5', 'r') as hf:
#     X_train = hf['X_train'][:]
# with h5py.File('X_test.h5', 'r') as hf:
#     X_test = hf['X_test'][:]
# with h5py.File('y_train.h5', 'r') as hf:
#     y_train = hf['y_train'][:]
# with h5py.File('y_test.h5', 'r') as hf:
#     y_test = hf['y_test'][:]

# print(type(X_train))
# print(X_train.shape)
# print(X_test.shape)
# print()
# print(X_train[0])
# print()
# print(y_train[0])
# exit()




# dic_action_transitions = {} 



# dic_shap_perEffect_perTrans_perAction = {}

# for num_action in range(0, 22):

#     dic_action_transitions["action_"+str(num_action)] = []

#     dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)] = {}

#     #with open("shap_vals_persisting_effects_removed/action_"+str(num_action)+".txt", "r") as file:
#     with open("shap_vals_persisting_effects_removed/action_"+str(num_action)+"_withEmptySet.txt", "r") as file:

#         for line in file:

#             if "transition" in line:
#                 last_key = int(line.split(" ")[1].strip())
#                 dic_action_transitions["action_"+str(num_action)].append(last_key)
#                 dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)][last_key] = {}


#             elif "add_" in line or "del_" in line:
#                 if len(dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)][last_key].values()) == 0:
#                     dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)][last_key] = {}
#                 dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)][last_key][line.split(" ")[0].strip()] = float(line.split(" ")[1].strip())





x_path = 'X.txt'
y_path = 'Y.txt'



# Load X (features)
X = pd.read_csv(x_path, sep=" ", header=None)

# Load Y (labels)
Y = pd.read_csv(y_path, sep=" ", header=None)
Y.name = "Label"







############################# START



preconds_names = ["(z"+str(i)+")" for i in range(50)]
neg_preconds_names = ["(not (z"+str(i)+"))" for i in range(50)]

preconds_names.extend(neg_preconds_names)




pos_preconds = np.loadtxt("action_pos4_BAK.csv", delimiter=' ', dtype=int)
neg_preconds = np.loadtxt("action_neg4_BAK.csv", delimiter=' ', dtype=int)

preconds_perEff_perAction = {}

for num_action in range(0, 22):

    preconds_perEff_perAction[num_action] = {}


    ########################################################
    #### Building the X   ##################################
    #### now considering ONE action (0)
    ########################################################

    action_transitions_preconds = []


    # # doing the union of all preconditions for this action
    # # (actually, for each transition representing this action)
    # for trans_id in dic_action_transitions["action_"+str(num_action)]:
    #     preconds_union = []
    #     preconds_union.extend(pos_preconds[trans_id])
    #     preconds_union.extend(neg_preconds[trans_id])
    #     action_transitions_preconds.append(preconds_union)

    # action_transitions_preconds of shape (48, 100)
    # i.e. pour chque transition, le vecteur concaténé des preconds pos et neg
    action_transitions_preconds = np.array(action_transitions_preconds)


    print(action_transitions_preconds.shape)




    #### Building the Y

    # retrieve all the unique effects names, put them as key in a dict

    # total set of effects for the high lvl action (here action 0)
    effects_set = []
    for trans_id, dico_effects in dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)].items():

        for eff_name, shap_val in dico_effects.items():
            if (eff_name not in effects_set): # and shap_val > 0 :
                effects_set.append(eff_name)


    # set of the effects, sorted alphabetically and by index (e.g. add_0 then add_1 etc)
    effects_set = sorted(effects_set, key=lambda x: (x.split('_')[0], int(x.split('_')[1])))



    # Making the table where ROWS = transition preconditions,   COL = SHAP of effects
    list_transIdKey_shapEffectsValues = [] # (#transitions, #effects)
    for j, (trans_id, dico_effects) in enumerate(dic_shap_perEffect_perTrans_perAction["action_"+str(num_action)].items()):

        #list_transIdKey_shapEffectsValues[j] = np.zeros(len(effects_set))

        list_transIdKey_shapEffectsValues.append([0 for ii in range(len(effects_set))])

        for i, effName in enumerate(effects_set):

            
            if effName in dico_effects.keys():
                if dico_effects[effName] > 0:
                    list_transIdKey_shapEffectsValues[j][i] = 1


    print("list_transIdKey_shapEffectsValueslist_transIdKey_shapEffectsValues")
    list_transIdKey_shapEffectsValues = np.array(list_transIdKey_shapEffectsValues)



    X = action_transitions_preconds

    print("X   ")
    print(X.shape) # (48, 100)

    print("XX HEAD")
    print(X[:10])

    

    Y = list_transIdKey_shapEffectsValues

    np.savetxt("X.txt", X, delimiter=" ", fmt="%d")
    np.savetxt("Y.txt", Y, delimiter=" ", fmt="%d")
    # Load X (features)
    X = pd.read_csv("X.txt", sep=" ", header=None)
    # Load Y (labels)
    Y = pd.read_csv("Y.txt", sep=" ", header=None)
    Y.name = "Label"

    # save_dataset("./", X, Y)

    # print(X.shape)
    # print(Y.shape)

    #### numpy to pandas


    # Rename columns
    new_columns = []
    for i in range(50):
        new_columns.append(f"z_{i}")
    for i in range(50):
        new_columns.append(f"not(z_{i})")

    X.columns = new_columns[:len(X.columns)] # Use slicing to handle cases where X might have fewer columns



    ############## DOING THE NEW X (and transo the Y and X into Pandas) ##################

    # Rename columns in Y
    new_y_columns = []
    for i in range(28):
        new_y_columns.append(f"add_{i}")
    for i in range(28):
        new_y_columns.append(f"not(add_{i})")

    Y.columns = new_y_columns[:len(Y.columns)] # Use slicing to handle cases where Y might have fewer columns




    # Create a new DataFrame for the modified X matrix
    new_X = pd.DataFrame()

    # Iterate through the first 50 columns of the original X
    for i in range(50):
        # Create a new column in new_X
        new_column = []
        for index, row in X.iterrows():
            if row[f'z_{i}'] == 1:
                new_column.append(1)
            elif row[f'not(z_{i})'] == 1:
                new_column.append(0)
            else:
                new_column.append('?')  # Or any other representation you prefer for "otherwise"
        new_X[f'z_{i}'] = new_column  # Assign the new column to the new DataFrame

    print("NEW X HEAD")
    print(new_X.head())
    new_X.head(2).to_csv('SEE_new_X_firstTwo.txt', sep='\t', index=False)
    question_marks_count = (new_X.iloc[0] == '?').sum()
    print("question_marks_count first row {}".format(str(question_marks_count)))
    
    exit()

    import numpy as np
    import itertools
    import pyarrow as pa
    import pyarrow.feather as feather

    # # Sample data creation for demonstration (replace with your actual data)
    # X = pd.DataFrame(np.random.choice(['0', '1', '?'], size=(48, 50), p=[0.45, 0.45, 0.1]))
    # Y = pd.DataFrame(np.random.rand(48, 56))

    print(new_X.shape)
    print(Y.shape)
    print(new_X[:2])
    print(Y[:2])

    # def expand_XY(X, Y):
    #     expanded_X = []
    #     expanded_Y = []

    #     for i in range(len(X)):
    #         print(f"Processing row {i + 1} of {len(X)}...")
    #         row_X = X.iloc[i]
    #         row_Y = Y.iloc[i]


    #         # Find positions of '?'
    #         question_mark_indices = [idx for idx, val in enumerate(row_X) if val == '?']
            
  
    #         # If there are no '?' in the row, just add the row as is
    #         if not question_mark_indices:
    #             expanded_X.append(row_X.values.tolist())
    #             expanded_Y.append(row_Y.values.tolist())
    #             continue
            
    #         # Generate all binary combinations for the '?' positions
    #         combinations = list(itertools.product('01', repeat=len(question_mark_indices)))

    #         # Replace '?' with each combination and duplicate rows
    #         for combination in combinations:
    #             new_row_X = row_X.copy()
    #             for idx, value in zip(question_mark_indices, combination):
    #                 new_row_X.iloc[idx] = int(value)
    #             expanded_X.append(new_row_X.values.tolist())
    #             expanded_Y.append(row_Y.values.tolist())
        
    #     # Convert back to DataFrame
    #     expanded_X_df = pd.DataFrame(expanded_X, columns=X.columns)
    #     expanded_Y_df = pd.DataFrame(expanded_Y, columns=Y.columns)
        
    #     return expanded_X_df, expanded_Y_df

    # Initialize global variables for progress tracking
    progress_counter = None
    lock = None

    def init_globals(counter, counter_lock):
        global progress_counter, lock
        progress_counter = counter
        lock = counter_lock


    def process_row(args):
        global progress_counter, lock
        row_X, row_Y, columns_X, columns_Y = args

        # Find positions of '?'
        question_mark_indices = [idx for idx, val in enumerate(row_X) if val == '?']

        # If there are no '?' in the row, just add the row as is
        if not question_mark_indices:
            with lock:
                progress_counter.value += 1
                print(f"Processed {progress_counter.value}")
            return [(row_X, row_Y)]

        # Generate all binary combinations for the '?' positions
        combinations = list(itertools.product('01', repeat=len(question_mark_indices)))

        expanded_rows = []

        # Progress bar for combinations
        with tqdm(total=len(combinations), desc="Row progress", dynamic_ncols=True, leave=False) as pbar:
            for combination in combinations:
                new_row_X = row_X.copy()
                for idx, value in zip(question_mark_indices, combination):
                    new_row_X[idx] = int(value)
                expanded_rows.append((np.array(new_row_X, dtype=np.int8), np.array(row_Y, dtype=np.int8)))
                pbar.update(1)  # Update progress bar

        # Increment the progress counter and print the current progress
        with lock:
            progress_counter.value += 1
            print(f"Processed {progress_counter.value}")
        

        # print("THE TYOPE")
        # print(type(expanded_rows))
        # print(type(expanded_rows[0]))
        # exit()
        return expanded_rows

    def expand_XY_parallel(X, Y):

        

        num_processes = cpu_count()
        total_rows = len(X)

        # Prepare the input rows
        rows = [(X.iloc[i].tolist(), Y.iloc[i].tolist(), X.columns, Y.columns) for i in range(total_rows)]


        # Initialize shared progress counter and lock
        global progress_counter, lock
        progress_counter = Value('i', 0)
        lock = Lock()

        # Create a Pool of workers and initialize global variables
        with Pool(processes=num_processes, initializer=init_globals, initargs=(progress_counter, lock)) as pool:
            results = pool.map(process_row, rows)

        print("BEFORE LISTING")

        # Flatten the list of results
        #expanded_X = []
        expanded_X = np.empty((0, X.shape[1]))
        #expanded_Y = []
        expanded_Y = np.empty((0, Y.shape[1]))
        for ijijijij, result in enumerate(results):

            print("doing the {} / {}".format(str(ijijijij), str(len(results))))
            for expanded_row_X, expanded_row_Y in result:

                expanded_row_X = np.expand_dims(expanded_row_X, axis=0)
                expanded_row_Y = np.expand_dims(expanded_row_Y, axis=0)

                # print("shape expanded_X {}".format(str(expanded_X.shape)))
                # print("shape expanded_row_X {}".format(str(expanded_row_X.shape)))

                expanded_X = np.concatenate((expanded_X, expanded_row_X), axis=0)
                expanded_Y = np.concatenate((expanded_Y, expanded_row_Y), axis=0)


                # expanded_X.append(expanded_row_X)
                # expanded_Y.append(expanded_row_Y)

        print("AFTER LISTING")

        

        # # Convert back to DataFrame
        # expanded_X_df = pd.DataFrame(expanded_X, columns=X.columns)
        # expanded_Y_df = pd.DataFrame(expanded_Y, columns=Y.columns)

        print("AFTER DATAFRAME CONVERSION")

        return expanded_X_df, expanded_Y_df




    # print(small)
    # small.to_csv('SEE_small.txt', sep='\t', index=False)
    # print("secondddd")
    # print(small.iloc[0]) # 0   0   1   ?    0   ?    1   0   0   ?   0    0    0    ? 
    # small.iloc[0].to_csv('SEE_smallILOC.txt', sep='\t', index=False)
    # exit()
    #print(new_X[:1].iloc[0])
    # z_0     0
    # z_1     1
    # z_2     ?
    # z_3     0

    #Example usage (ensure you have X and Y as pandas DataFrames):

    # print(new_X.values.dtype)
    # print(Y.values.dtype)

    # print(new_X.shape)
    # print()
    # print(Y.shape)
    # exit()

    #expanded_X, expanded_Y = expand_XY_parallel(new_X, Y)
    #expanded_X, expanded_Y = expand_XY(new_X, Y)


    # #print(expanded_X.head(16))
    # new_X[:1].to_csv('SEE_newX.txt', sep='\t', index=False)
    # expanded_X.to_csv('SEE_expanded_X.txt', sep='\t', index=False)
    # expanded_Y.to_csv('SEE_expanded_Y.txt', sep='\t', index=False)
    # exit()



    # # Save expanded data using pyarrow for efficiency
    # feather.write_feather(pa.Table.from_pandas(expanded_X), 'expanded_X.feather')
    # feather.write_feather(pa.Table.from_pandas(expanded_Y), 'expanded_Y.feather')

    # Here we retrieve the Feathers (coz expand_XY is so long)
    expanded_X = pd.read_feather('expanded_X.feather')
    expanded_Y = pd.read_feather('expanded_Y.feather')


    print("expanded_X shape") # (33607680, 50)
    print(expanded_X.shape)
    print("expanded_Y shape") # (33607680, 56)
    print(expanded_Y.shape)

    exit()
    # print("Y0")
    # print(type(expanded_Y))
    #expanded_X = expanded_X.to_numpy()
    # print("Y0")
    # print(expanded_Y[0])
    # print("Y1")
    # print(expanded_Y[1])


    # # Convert each row to a tuple (hashable type) for fast uniqueness computation
    # vector_tuples = map(tuple, expanded_X)
    # print("X2")
    # # Use a set to find unique rows
    # unique_vectors = set(vector_tuples)
    # print('X3')
    # # Count of unique vectors
    # num_unique_vectors = len(unique_vectors)

    # print(f"Number of unique vectors: {num_unique_vectors}")
    # exit()

    # # Select a subset of rows (e.g., first 100 rows)
    # subset_X = expanded_X.iloc[:20]  # Adjust the number as needed
    # subset_Y = expanded_Y.iloc[:20]  # Adjust the number as needed
    # # Save the subsets back to Feather files
    # subset_X.to_feather('subset_X.feather')
    # subset_Y.to_feather('subset_Y.feather')



    # expanded_X = pd.read_feather('subset_X.feather')
    # expanded_Y = pd.read_feather('subset_Y.feather')


    # print("some shapes")

    # print(expanded_X.shape) # (20, 50)
    # # print(expanded_Y.shape) # (20, 56)


    # # Here we TRAIN the DT
    # X = expanded_X.values
    # Y = expanded_Y.values



    # import time
    # start_time = time.time()
    # print("laaaa")
    # Diviser les données en ensembles d’entraînement et de test pour valider le modèle
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(expanded_X, expanded_Y, test_size=0.2, random_state=42)
    #print("--- %s seconds ---" % (time.time() - start_time))

    import h5py

    with h5py.File('X_train.h5', 'w') as hf:
        hf.create_dataset('X_train', data=X_train)

    with h5py.File('X_test.h5', 'w') as hf:
        hf.create_dataset('X_test', data=X_test)

    with h5py.File('y_train.h5', 'w') as hf:
        hf.create_dataset('y_train', data=y_train)

    with h5py.File('y_test.h5', 'w') as hf:
        hf.create_dataset('y_test', data=y_test)


    exit()
    # with h5py.File('X_train.h5', 'r') as hf:
    #     X_train = hf['X_train'][:]
    # with h5py.File('X_test.h5', 'r') as hf:
    #     X_test = hf['X_test'][:]
    # with h5py.File('y_train.h5', 'r') as hf:
    #     y_train = hf['y_train'][:]
    # with h5py.File('y_test.h5', 'r') as hf:
    #     y_test = hf['y_test'][:]







    #####################################




    # Function to estimate total nodes
    def estimate_total_nodes(n_estimators, num_leaves, num_labels):
        # Total nodes = n_estimators * num_leaves per tree * number of labels
        return n_estimators * num_leaves * num_labels

    # # Example data
    # np.random.seed(42)
    # X = np.random.rand(100, 10)
    # y = np.random.randint(0, 2, (100, 3))  # Multilabel problem with 3 labels

    # Split data
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # print(X_train.shape) #
    # print(y_train.shape) #
    # print()
    # print(X_test.shape) # 
    # print(y_test.shape) #

    # # Parameters
    num_leaves = 100 #33607680 #41 #31
    n_estimators = 1 # 100

    # # Calculate the total number of nodes
    # num_labels = y.shape[1]  # Number of labels
    # total_nodes = estimate_total_nodes(n_estimators, num_leaves, num_labels)

    # # Initialize iteration logger with total nodes

    ### Initialize LightGBM classifier
    base_classifier = LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=num_leaves,
        learning_rate=0.1,
        n_estimators=n_estimators,
        verbose=-1
    )

    # calcule le nbre de vecteurs différents

    #### Wrap LightGBM in a Classifier Chain
    classifier = ClassifierChain(classifier=base_classifier)

    # Train Classifier Chain with progress logging
    # LightGBM's internal callback mechanism handles the progress tracking
    classifier.fit(X_train, y_train, log_progress=True)



    

    # Save the trained classifier to a file
    joblib.dump(classifier, "classifier_chain_model.pkl")
    print("Model saved successfully!")
    # Load the classifier from the file
    # classifier = joblib.load("classifier_chain_model.pkl")
    # print("Model loaded successfully!")

    # Use ClassifierChain for prediction
    y_pred = classifier.predict(X_test).toarray()  # Use classifier.predict(), not base_classifier.predict()

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    hamming = hamming_loss(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Hamming Loss: {hamming}")


    exit()


    print(classifier.classifiers_)


    # exit()

    feature_names_part_1 = [f"(z{i})" for i in range(50)]
    feature_names_part_2 = [f"not (z{i})" for i in range(50)]

    feature_names = feature_names_part_1
    feature_names.extend(feature_names_part_2)



    # # Determine grid size (square layout)
    # num_trees = len(classifier.classifiers_)
    # grid_size = math.ceil(math.sqrt(num_trees))  # Closest square layout

    # # Create a figure large enough to hold all subplots
    # fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    # axes = axes.flatten()  # Flatten to iterate easily


    #ax = lgb.plot_tree(model, tree_index=0, figsize=(20, 10))

    # Access models from the ClassifierChain
    for i, (estimator, effect_name) in enumerate(zip(classifier.classifiers_, effects_set)):



        y_pred_last_tree = estimator.predict(X_test, start_iteration=99, num_iteration=1)
        accuracy = accuracy_score(y_test, y_pred_last_tree)
        print(f"Accuracy using the last tree: {accuracy}")

        exit()

        # Check each tree's structure
        for i in range(booster.num_trees()):
            tree_info = booster.dump_model()["tree_info"][i]
            tree_structure = tree_info["tree_structure"]
            if "split_feature" in tree_structure:
                print(f"Tree {i} has splits.")
            else:
                print(f"Tree {i} has no splits (single leaf).")



        ax = lgb.plot_tree(estimator, tree_index=0, figsize=(20, 10))

        plt.savefig("lgbm_tree_plot_"+str(i)+".png", dpi=300, bbox_inches="tight")  # Save as PNG
        exit()
        # ax = axes[i]
        # plot_tree(estimator, 
        #         feature_names=feature_names, 
        #         label='all',
        #         class_names=[f'Class {k}' for k in range(2)],  # Assuming binary classification for each output
        #         filled=False, rounded=True,
        #         impurity=False,
        #         ax=ax)
        # ax.set_title(f"Effect: {effect_name}", fontsize=8)



        # print(f"Rules for Label {i + 1}:")
        # # Export the tree structure for the LightGBM model as text
        # tree_rules = model.booster_.dump_model()
        # print(f"Tree structure for Label {i + 1}:\n", tree_rules)
        # # Save the tree structure to a file (optional)
        # with open(f"label_{i + 1}_tree.json", "w") as f:
        #     import json
        #     json.dump(tree_rules, f)




    # # Hide any extra subplots (if there are fewer trees than grid spaces)
    # for j in range(i + 1, len(axes)):
    #     fig.delaxes(axes[j])

    # # Adjust layout for better spacing
    # plt.tight_layout()

    # # Save to PDF
    # output_pdf_file = "DT_LightGBM.pdf"



    # fig.savefig(output_pdf_file, format="pdf", dpi=300)
    # plt.close(fig)

    # print(f"All decision trees saved into {output_pdf_file} in a square layout")




    exit()
    ####################################################












    # # y_train = np.array(y_train).ravel()
    # # y_test = np.array(y_test).ravel()

    # # Prepare LightGBM datasets
    # train_data = lgb.Dataset(X_train, label=y_train)
    # test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)



    # # Parameters for LightGBM
    # params = {
    #     'objective': 'binary',
    #     'metric': 'binary_error',
    #     'boosting_type': 'gbdt',
    #     'num_leaves': 31,
    #     'max_depth': 3,
    #     'learning_rate': 0.1,
    #     'verbose': -1
    # }

    # # Estimate the total number of nodes tested per tree
    # def estimate_tested_nodes(num_features, num_bins, max_depth):
    #     return sum(2**i for i in range(max_depth)) * num_features * num_bins

    # # Parameters for estimation
    # num_features = X_train.shape[1]
    # num_bins = 255  # Default for LightGBM
    # max_depth = params.get('max_depth', -1)
    # if max_depth <= 0:  # Handle case where max_depth is not set
    #     max_depth = float('inf')

    # nodes_per_tree = estimate_tested_nodes(num_features, num_bins, max_depth)
    # print(f"Estimated nodes tested per tree: {nodes_per_tree}")

    # # Custom callback to track progress
    # class NodeProgressCallback:
    #     def __init__(self, total_nodes):
    #         self.total_nodes = total_nodes
    #         self.current_nodes = 0

    #     def __call__(self, env):
    #         self.current_nodes += nodes_per_tree
    #         progress = self.current_nodes / self.total_nodes * 100
    #         print(f"Progress: {progress:.2f}% ({self.current_nodes}/{self.total_nodes} nodes tested)")

    # # Total nodes for all trees
    # num_trees = 10
    # total_nodes = nodes_per_tree * num_trees
    # callback = NodeProgressCallback(total_nodes)

    # # Train the LightGBM model
    # model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=num_trees, callbacks=[callback])






















    exit()
    



    # NOW, once the DT has been TRAINED.... WHAT TO DO ???

    # LE DISPLAY



# Display the sizes of the new datasets for confirmation

# # Load the expanded data from HDF5 files
# expanded_X = pd.read_hdf('/path/to/expanded_X.h5', key='X')
# expanded_Y = pd.read_hdf('/path/to/expanded_Y.h5', key='Y')

# # Display the loaded data
# print(expanded_X.head())
# print(expanded_Y.head())


# import pandas as pd
# from tqdm import tqdm
# import csv
# import numpy as np

# # Function to duplicate rows and save to file using NumPy and optimized writing
# def duplicate_rows_to_file_numpy_optimized(df_x, df_y, x_file_path, y_file_path):

#     with open(x_file_path, 'w', newline='') as x_file, \
#             open(y_file_path, 'w', newline='') as y_file:

#         x_writer = csv.writer(x_file)
#         y_writer = csv.writer(y_file)

#         # Write header rows
#         x_writer.writerow(df_x.columns)
#         y_writer.writerow(df_y.columns)

#         # Convert DataFrames to NumPy arrays for faster processing
#         X_np = df_x.to_numpy()
#         Y_np = df_y.to_numpy()

#         # Identify rows with '?' and rows without '?'
#         unknown_rows_mask = np.any(X_np == '?', axis=1)
#         known_rows_mask = ~unknown_rows_mask

#         # Write known rows directly to files
#         x_writer.writerows(X_np[known_rows_mask].tolist())
#         y_writer.writerows(Y_np[known_rows_mask].tolist())

#         # Process rows with '?' using NumPy
#         unknown_rows = X_np[unknown_rows_mask]
#         unknown_rows_y = Y_np[unknown_rows_mask]

#         for i, row in enumerate(tqdm(unknown_rows, total=len(unknown_rows), desc="Processing unknown rows")):
#             unknown_cols_indices = np.where(row == '?')[0]
#             num_unknowns = len(unknown_cols_indices)

#             # Generate combinations using NumPy
#             combinations = np.array(np.meshgrid(*[[0, 1]] * num_unknowns)).T.reshape(-1, num_unknowns)

#             # Create new rows using broadcasting and advanced indexing
#             new_rows = np.repeat(row[np.newaxis, :], len(combinations), axis=0)
#             new_rows[:, unknown_cols_indices] = combinations

#             # Write new rows to files
#             x_writer.writerows(new_rows.tolist())
#             y_writer.writerows([unknown_rows_y[i].tolist()] * len(combinations))



# # Define output file paths
# x_file_path = 'new_X.csv'
# y_file_path = 'new_Y.csv'

# # Apply the function to save the data
# duplicate_rows_to_file_numpy_optimized(new_X, Y, x_file_path, y_file_path)

# # # Load the compressed files as NumPy arrays with minimal memory usage
# # new_X = np.loadtxt('new_X.csv', delimiter=',', dtype=np.int8, skiprows=1) # Assuming the first row is a header
# # new_Y = np.loadtxt('new_Y.csv', delimiter=',', dtype=np.int8, skiprows=1) # Assuming the first row is a header

# # print(new_X.shape)
# # new_Y.shape