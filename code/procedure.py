"""from graphviz import Digraph

g = Digraph('G', filename='annotation_procedure_3.gv')

# Add nodes with updated labels
g.node('A', 'Read the entire text')
g.node('B', 'For each lexical unit:', shape='doublecircle')  # Highlighting node 'B'
g.node('C', '   Is it (A) ironic, (B) metaphorical?'
g.node('D', '     If yes, annotate and guess/look up intended meaning')
g.node('E', '     If no, assume basic meaning')
g.node('F', '   Does the unit construct a qualitative or quantitative scale?')
g.node('G', '     Determine the ontological referent')
g.node('H', '     Is it more extreme than justified?')
g.node('I', '       If yes, it is hyperbolic')
g.node('J', '       If no, move to next lexical unit')

# Add edges
g.edge('A', 'B')
g.edges(['BC', 'BF', 'BG', 'BH'])
g.edge('C', 'D', label='Composite')
g.edge('c', 'e', label='Basic')
g.edge('F', 'J', label='Non-hyperbolic')
g.edge('H', 'I', label='Hyperbolic')
g.edge('H', 'J', label='Non-hyperbolic')

# Render the graph to a file
g.view()
"""

from graphviz import Digraph

g = Digraph('G', filename='annotation_procedure_3.gv')

# Reading process cluster
with g.subgraph(name='cluster_0') as c:
    c.attr(style='filled', color='lightgrey')
    c.node_attr.update(style='filled', color='white')
    c.node('A', 'Read the entire text')
    c.node('B', 'For each lexical unit:')  # Highlighting node 'B'
    c.attr(label='Reading Process')

# Auxiliary figures cluster
with g.subgraph(name='cluster_1') as c:
    c.attr(style='filled', color='lightyellow')
    c.node_attr.update(style='filled', color='white')
    c.node('C', 'Is it (A) ironic, (B) metaphorical?')
    c.node('D', 'If any, annotate and guess/look up intended meaning')
    c.node('E', 'If none, assume basic meaning')
    c.edges([('C', 'D'), ('C', 'E')])
    c.attr(label='Auxiliary figures of speech')

# Hyperbole determination cluster
with g.subgraph(name='cluster_2') as c:
    c.attr(style='filled', color='lightblue')
    c.node_attr.update(style='filled', color='white')
    c.node('F', 'Does the unit construct a qualitative or quantitative scale?')
    c.node('G', 'Determine the ontological referent')
    c.node('H', 'Is the unit more extreme than justified given referent?')
    c.node('I', 'If yes, it is hyperbolic')
    c.node('J', 'If no, move to next lexical unit')
    c.edges([('F', 'G'), ('G', 'H'), ('H', 'I'), ('H', 'J')])
    c.attr(label='Hyperbole Determination')

# Add edges between clusters
g.edge('A', 'B')
g.edge('B', 'C')
g.edge('B', 'F')
g.edge('E', 'J')
g.edge('I', 'J')

# Render the graph to a file
g.view()