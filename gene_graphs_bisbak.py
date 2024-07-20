import matplotlib.pyplot as plt

dico = {
    "blocks": {
        "r_latplan": {1: 3, 3: 3, 6: 3, 12: 3},
        "vanilla": {1: 3, 3: 2, 6: 0, 12: 0}
    },
    "sokoban": {
        "r_latplan": {1: 3, 5: 3, 10: 3, 18: 3},
        "vanilla": {1: 3, 5: 2, 10: 0, 18: 0}
    },
    "hanoi": {
        "r_latplan": {1: 3, 3: 3, 5: 3, 8: 3},
        "vanilla": {1: 3, 3: 2, 5: 0, 8: 0}
    }
}

# Plotting the results on the same plot
plt.figure(figsize=(12, 8))

for domain, models in dico.items():
    for model, results in models.items():
        x = list(results.keys())
        y = list(results.values())
        plt.plot(x, y, marker='o', label=f"{domain} - {model}")

plt.title("Results of Experiments")
plt.xlabel("Problem Size")
plt.ylabel("Results")
plt.legend()
plt.grid(True)

plt.savefig('plan_perfs_hanoi.png')
