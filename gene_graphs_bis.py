import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines


#  EXP 1 , 2
dico = {
    "blocks": {
        "r_latplan": {1.25: 98, 3: 98, 6: 98, 12: 98},
        "vanilla": {1.25: 98, 3: 76, 6: 2, 12: 2}
    },
    "sokoban": {
        "r_latplan": {1.35: 96, 5: 96, 10: 96, 18: 96},
        "vanilla": {1.35: 96, 5: 75, 10: 4, 18: 4}
    },
    "hanoi": {
        "r_latplan": {1: 100, 3: 100, 5: 100, 8: 100},
        "vanilla": {1: 100, 3: 74, 5: 0, 8: 0}
    }
}


# Define colors for each domain
colors = {
    "blocks": 'blue',
    "sokoban": 'green',
    "hanoi": 'red'
}

# Plotting the results on the same plot
plt.figure(figsize=(12, 4))

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
plt.ylabel("% solved")
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# Set y-axis ticks from 0 to 100 with a step of 10
plt.yticks(range(0, 101, 10))

# Custom legend entries
r_latplan_line = mlines.Line2D([], [], color='black', linestyle='-', label="R-Latplan \n& RC-Latplan")
vanilla_line = mlines.Line2D([], [], color='black', linestyle='--', label='Vanilla')
blocks_line = mlines.Line2D([], [], color='blue', linestyle='-', label='blocks')
sokoban_line = mlines.Line2D([], [], color='green', linestyle='-', label='sokoban')
hanoi_line = mlines.Line2D([], [], color='red', linestyle='-', label='hanoi')

# Adding custom legend
domain_legend = plt.legend(handles=[blocks_line, sokoban_line, hanoi_line], loc='lower left', bbox_to_anchor=(0, 0), frameon=False)
model_legend = plt.legend(handles=[r_latplan_line, vanilla_line], loc='lower left', bbox_to_anchor=(0, 0.2), frameon=False)
plt.gca().add_artist(domain_legend)

plt.savefig('plan_perfs_hanoi_BIS_222222.png')

