#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for Network Plot

This will export a file 'out.png' in your working directory
"""

        
# Plotting the classes network based on the distance matrix 
if __name__ == '__Main__':
    from Main import centers
    
import networkx as nx
from scipy.spatial import distance_matrix
dt = [('len', float)]
A = distance_matrix(centers,centers)
A = A.view(dt)

labels = {}
for i in range(10):
    labels[i] = "C%i"%i

G = nx.from_numpy_matrix(A)
G = nx.relabel_nodes(G, labels)    

G = nx.drawing.nx_agraph.to_agraph(G)

G.node_attr.update(color="red", style="filled")
G.edge_attr.update(color="blue", width="2.0")

G.draw('out.png', format='png', prog='neato')