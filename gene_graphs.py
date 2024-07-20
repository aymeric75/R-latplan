import matplotlib.pyplot as plt

# Data dictionaries SOKOBAN
# dico_vanilla = {
#     1: 3,
#     5: 1,
#     10: 0,
#     18: 0
# }

# dico_r_latplan = {
#     1: 3,
#     5: 3,
#     10: 3,
#     18: 3
# }

# DICO BLOCKS

dico_vanilla = {
    1: 3,
    3: 2,
    5: 0,
    8: 0
}

dico_r_latplan = {
    1: 3,
    3: 3,
    5: 3,
    8: 3,
}



dico = {

    "blocks": {
        "r_latplan": {
            1: 3,
            3: 3,
            6: 3,
            12: 3
        },
        "vanilla": {
            1: 3,
            3: 2,
            6: 0,
            12: 0
        }
    },

    "sokoban": {
        "r_latplan": {
            1: 3,
            5: 3,
            10: 3,
            18: 3
        },
        "vanilla": {
            1: 3,
            5: 2,
            10: 0,
            18: 0
        }
    },
    "hanoi": {
        "r_latplan": {
            1: 3,
            3: 3,
            5: 3,
            8: 3
        },
        "vanilla": {
            1: 3,
            3: 2,
            5: 0,
            8: 0
        }
    }
}


# Extracting keys and values for plotting
x_vanilla, y_vanilla = zip(*dico_vanilla.items())
x_r_latplan, y_r_latplan = zip(*dico_r_latplan.items())



def turn_int(liste):
    liste_ = list(liste)
    for i, e in enumerate(liste):
        
        liste_[i] = int(e)
    return liste_ 


x_vanilla = turn_int(x_vanilla)
y_vanilla = turn_int(y_vanilla)
x_r_latplan = turn_int(x_r_latplan)
y_r_latplan = turn_int(y_r_latplan)

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(x_vanilla, y_vanilla, label='Vanilla', marker='o', color="red")
plt.plot(x_r_latplan, y_r_latplan, label='R Latplan', marker='s', color="blue")

# Adding titles and labels
plt.title('Planning performances on Hanoi: Vanilla vs R-Latplan')
plt.xlabel('Problem size (in steps)')
plt.ylabel('Number of problems solved')
plt.legend()
plt.grid(True)

plt.savefig('plan_perfs_hanoi.png')

# # Display the plot
# plt.show()
