import numpy as np


######  the fnal_set , i.e. what we want to put in the OR
candidates = np.array([
    [0,0,0,1,0],
    [1,0,1,0,0],
    [0,0,0,0,1],
    [0,1,0,0,0],
])


##### s_orig, i.e. the preconditions sets
targets = np.array([
    [1,0,1,0,1],
    [0,0,1,0,0],
    [1,1,1,0,0]
])


def count_targets_entailing_each_candidate(candidates, targets):
    """
    Returns a 1D array of length num_candidates,
    where each value is the number of targets that entail that candidate.
    """
    num_candidates = candidates.shape[0]
    num_targets = targets.shape[0]

    # Expand dims for broadcasting: (num_targets, 1, 32) & (1, num_candidates, 32)
    # Resulting shape: (num_targets, num_candidates, 32)
    comparison = (targets[:, None, :] & candidates[None, :, :]) == candidates[None, :, :]

    # Now check for all bits matching across the last axis
    entailed_matrix = np.all(comparison, axis=2)  # shape: (num_targets, num_candidates)

    # Count for each candidate how many targets entail it
    count_per_candidate = np.sum(entailed_matrix, axis=0)  # shape: (num_candidates,)


    return np.argmax(count_per_candidate)
    

#return count_per_candidate


counts = count_targets_entailing_each_candidate(candidates, targets)

print(np.argmax(counts))

exit()

for i, count in enumerate(counts):
    print(f"Candidate {i} is entailed by {count} targets.")