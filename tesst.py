
def load_dataset(path_to_file):
    import pickle
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    # print("path_to_file path_to_file")
    # print(path_to_file)
    # exit()
    return loaded_data


# data_datasets = load_dataset("data_datasets.p")

# print(type(data_datasets))
# print(data_datasets.keys())
# <class 'dict'>
# dict_keys(['train_set', 'test_val_set', 'all_pairs_of_images_reduced_orig', 'all_actions_one_hot', 'all_high_lvl_actions_one_hot', 'mean_all', 'std_all', 'all_actions_unique', 'all_high_lvl_actions_unique', 'orig_max', 'orig_min', 'train_set_no_dupp_processed', 'train_set_no_dupp_orig', 'all_traces_pair_and_action'])

# data_exps = load_dataset("data_exps.p")

# print(type(data_exps))
# print(data_exps.keys())