import pickle


def load_dataset(path_to_file):
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data




loaded = load_dataset("traces.p")
print(loaded.keys())
print(type(loaded))

print(len(loaded["unique_transitions"]))

print(type(loaded["unique_obs_img"]))