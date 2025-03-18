import pandas as pd
import matplotlib.pyplot as plt

# Define table data
columns = ["", "x_all_atoms_dt2", "x_atoms_from_cond_effects_dt2", "x_atoms_outside_of_cond_effects_dt2"]
index = ["", "", "filter_out_dt1", "", "no_filter_out_dt1", ""]
data = [
    ["", "", "", ""],
    ["", "filter_out_dt2", "no_filter_out_dt2", "filter_out_dt2", "no_filter_out_dt2"],
    ["factorize_dt1", "", "", ""],
    ["no_factorize_dt1", "", "", ""],
    ["factorize_dt1", "", "", ""],
    ["no_factorize_dt1", "", "", ""]
]

# Create DataFrame
df = pd.DataFrame(data, columns=columns, index=index)

# Plot table
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 rowLabels=df.index,
                 cellLoc='center',
                 loc='center')

# Save table as an image
plt.savefig("table_output.png", bbox_inches='tight')
plt.close()