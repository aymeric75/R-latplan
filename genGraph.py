# from PySimpleAutomata import DFA, automata_IO
from networkx.drawing.nx_pydot import write_dot
# #dfa_example = automata_IO.dfa_json_importer("DFA_blocks4Colors.json")
# #dfa_example = automata_IO.dfa_dot_importer("DFA-Hanoi_4_4.dot")
# dfa_example = automata_IO.dfa_json_importer("DFA_Hanoi_4_4__.json")

# new_dfa=DFA.dfa_reachable(dfa_example)
# # new_dfa=DFA.dfa_completion(dfa_example)
# # print(new_dfa)
# # exit()
# # engine='neato'
# automata_IO.dfa_to_dot(new_dfa, 'Full_DFA_hanoi_4-4', './', engine='neato')


# # dfa_completion(dfa)


# from PySimpleAutomata import DFA, automata_IO

# #dfa_example = automata_IO.dfa_json_importer("DFA_blocks4Colors.json")
# #dfa_example = automata_IO.dfa_dot_importer("DFA-Hanoi_4_4.dot")
# dfa_example = automata_IO.dfa_json_importer("Full_DFA_Sokoban_6_6.json")

# new_dfa=DFA.dfa_reachable(dfa_example)
# # new_dfa=DFA.dfa_completion(dfa_example)
# # print(new_dfa)
# # exit()
# # engine='neato'
# automata_IO.dfa_to_dot(new_dfa, 'Full_DFA_Sokoban_6_6', './', engine='neato')


# # dfa_completion(dfa)

import json
import networkx as nx


with open('DFA_Hanoi_4_4__.json') as f:
    json_data = json.loads(f.read())

G = nx.DiGraph()

G.add_nodes_from(
    elem
    for elem in json_data['states']
)
G.add_edges_from(
    (elem[0], elem[1])
    for elem in json_data['transitions']
)

nx.draw(
    G,
    with_labels=True
)

#write_dot(G, "dfa.dot")

print(len(G.edges()))
print(len(G.nodes()))

write_dot(G, "DFA_Hanoi_4_4__.dot")