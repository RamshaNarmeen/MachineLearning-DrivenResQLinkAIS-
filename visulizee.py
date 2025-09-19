import networkx as nx
import matplotlib.pyplot as plt

# Create a graph representing devices/nodes in the mesh network
G = nx.Graph()

# Add nodes representing devices
nodes = ['A', 'B', 'C', 'D']
G.add_nodes_from(nodes)

# Add edges representing Bluetooth/Wi-Fi connections
edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'C')]
G.add_edges_from(edges)

# Example message path through the network
message_path = ['A', 'B', 'C', 'D']

# Position nodes in a layout for visualization
pos = nx.spring_layout(G, seed=42)  # fixed seed for consistent layout

# Draw all nodes and edges
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1000)
nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')

# Highlight the path of the delivered message
path_edges = list(zip(message_path, message_path[1:]))

nx.draw_networkx_edges(
    G, pos,
    edgelist=path_edges,
    width=4,
    edge_color='red'
)

plt.title("ResQLink-AIS Message Path")
plt.axis('off')
plt.show()
