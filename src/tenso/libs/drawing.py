import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

from tenso.state.pureframe import Frame


def visualize_frame(frame: Frame, fname='frame_graph_output'):
    G = nx.MultiGraph()
    graph = frame.get_graph()
    nodes = frame.nodes
    ends = frame.ends
    for node in nodes:
        G.add_node(node, shape='none')
    for end in ends:
        G.add_node(end, shape='underline')

    # add edges
    plotted = set()
    for p, cs in graph.items():
        for c in cs:
            if c in plotted:
                continue
            G.add_edge(p, c, len=1.5)
        plotted.add(p)

    A = to_agraph(G)
    #A.node_attr['fontname'] = 'Arial'

    A.draw(fname + '.pdf', format='pdf', prog='neato')

    return
