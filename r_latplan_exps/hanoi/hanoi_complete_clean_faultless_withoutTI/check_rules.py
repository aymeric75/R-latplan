from itertools import combinations

def detect_contradictions(rules):
    """
    Detect contradictions in a set of IF-THEN rules.
    
    :param rules: A list of tuples (condition, effect)
    :return: List of contradictions found
    """
    contradictions = []
    rule_dict = {}

    # Build rule dictionary where conditions map to sets of effects
    for condition, effect in rules:
        if condition not in rule_dict:
            rule_dict[condition] = set()
        rule_dict[condition].add(effect)

    # Check for contradictions within the same condition
    for condition, effects in rule_dict.items():
        for eff1, eff2 in combinations(effects, 2):
            if effects_are_conflicting(eff1, eff2):
                contradictions.append(f"Contradiction: IF {condition} THEN {eff1} vs IF {condition} THEN {eff2}")

    # Check for contradictions between mutually exclusive conditions
    conditions = list(rule_dict.keys())
    for cond1, cond2 in combinations(conditions, 2):
        if conditions_are_mutually_exclusive(cond1, cond2):
            common_effects = rule_dict[cond1] & rule_dict[cond2]
            if common_effects:
                contradictions.append(f"Inconsistency: {cond1} and {cond2} cannot be true together, yet they both imply {common_effects}")

    return contradictions

def effects_are_conflicting(eff1, eff2):
    """
    Checks if two effects are contradictory.
    
    :param eff1: Effect 1
    :param eff2: Effect 2
    :return: Boolean indicating if they conflict
    """
    return (f"(not {eff1})" == eff2) or (f"(not {eff2})" == eff1)

def conditions_are_mutually_exclusive(cond1, cond2):
    """
    Checks if two conditions are mutually exclusive.
    
    :param cond1: Condition 1
    :param cond2: Condition 2
    :return: Boolean indicating if they are mutually exclusive
    """
    return (f"(not {cond1})" == cond2) or (f"(not {cond2})" == cond1)

# Example rule set (each condition can have multiple effects)
rules = [
    ("(A)", "(X)"),
    ("(A)", "(Y)"),  # Allowed, A can lead to both X and Y
    ("(B)", "(Z)"),
    ("(A)", "(not (X))"),  # Contradiction: A -> X and A -> NOT X
    ("(not (A))", "(X)"),  # Inconsistency: A and NOT A shouldn't both imply X
]

contradictions = detect_contradictions(rules)

if contradictions:
    print("Contradictions found:")
    for c in contradictions:
        print(c)
else:
    print("No contradictions found.")
