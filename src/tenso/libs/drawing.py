"""A tool to draw graphs from the Frames
"""
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

from tenso.state.pureframe import Frame


def visualize_frame(frame: Frame, fname='frame_graph_output'):
    """Function to draw a simple representation of the graph structure of
    a given frame to a pdf file.

    :param frame: The :class: Frame whose graph is to be drawn
    :type frame: :class: Frame

    :param fname: The file name where the graph will be drawn, default 'frame_graph_output'
    :type fname: string
    """
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

    A.draw(fname + '.pdf', format='pdf', prog='neato')

    return
