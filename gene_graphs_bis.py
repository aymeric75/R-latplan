import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines


#  EXP 1 , 2
# dico = {
#     "blocks": {
#         "r_latplan": {1: 3.03, 3: 3.03, 6: 3.03, 12: 3.03},
#         "vanilla": {1: 3, 3: 2, 6: 0.03, 12: 0.03}
#     },
#     "sokoban": {
#         "r_latplan": {1: 3.06, 5: 3.06, 10: 3.06, 18: 3.06},
#         "vanilla": {1: 3, 5: 2, 10: 0.06, 18: 0.06}
#     },
#     "hanoi": {
#         "r_latplan": {1: 3, 3: 3, 5: 3, 8: 3},
#         "vanilla": {1: 3, 3: 2, 5: 0, 8: 0}
#     }
# }



dico = {
    "blocks": {
        "r_latplan": {, 13: 3.03},
        "vanilla": {1: 3, 3: 2, 6: 0.03, 12: 0.03}
    },
    "sokoban": {
        "r_latplan": {1: 3.06, 5: 3.06, 10: 3.06, 18: 3.06},
        "vanilla": {1: 3, 5: 2, 10: 0.06, 18: 0.06}
    },
    "hanoi": {
        "r_latplan": {1: 3, 3: 3, 5: 3, 8: 3},
        "vanilla": {1: 3, 3: 2, 5: 0, 8: 0}
    }
}



# Define colors for each domain
colors = {
    "blocks": 'blue',
    "sokoban": 'green',
    "hanoi": 'red'
}

# Plotting the results on the same plot
plt.figure(figsize=(12, 8))

for domain, models in dico.items():
    for model, results in models.items():
        x = list(results.keys())
        y = list(results.values())
        if model == "r_latplan":
            linestyle = '-'
        else:
            linestyle = '--'
        plt.plot(x, y, marker='o', linestyle=linestyle, color=colors[domain], label=f"{domain} - {model}")

plt.title("")
plt.xlabel("Problem Size")
plt.ylabel("#solved")
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))


# Custom legend entries
r_latplan_line = mlines.Line2D([], [], color='black', linestyle='-', label='R-latplan')
vanilla_line = mlines.Line2D([], [], color='black', linestyle='--', label='Vanilla')
blocks_line = mlines.Line2D([], [], color='blue', linestyle='-', label='blocks')
sokoban_line = mlines.Line2D([], [], color='green', linestyle='-', label='sokoban')
hanoi_line = mlines.Line2D([], [], color='red', linestyle='-', label='hanoi')

# Adding custom legend
domain_legend = plt.legend(handles=[blocks_line, sokoban_line, hanoi_line], loc='lower left', bbox_to_anchor=(0, 0), frameon=False)
model_legend = plt.legend(handles=[r_latplan_line, vanilla_line], loc='lower left', bbox_to_anchor=(0, 0.1), frameon=False)
plt.gca().add_artist(domain_legend)

plt.savefig('plan_perfs_hanoi.png')

