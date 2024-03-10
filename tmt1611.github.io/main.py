import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import random
from community import community_louvain
import numpy as np
from alphashape import alphashape
import datetime
import pandas as pd

def alpha_shape(coordinates, alpha=0):
    # Compute the alpha-shape
    alpha_shape = alphashape(coordinates, alpha)

    # Plot the alpha-shape
    x, y = alpha_shape.exterior.xy
    return list(x), list(y)

def create_network_graph(df):
    # Assuming you have a graph G
    G = nx.Graph()
    nodes = df['ID'].unique()
    G.add_nodes_from(nodes)

    edges = zip(df['parent'][1:], df['ID'][1:])
    G.add_edges_from(edges)

    pos = generate_pos(G)

    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node trace and get node positions
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_adjacencies = []
    node_text = []
    node_value = []
    node_value_num = []
    border_colors = []
    level = []

    for node, adjacencies in enumerate(G.adjacency()):
        node_name = list(G.nodes)[node]
        border_colors.append('black')
        node_value_num.append(df.loc[df['ID'] == node_name, 'value_num'].values[0])
        node_value.append(df.loc[df['ID'] == node_name, 'value'].values[0])
        node_adjacencies.append(len(adjacencies[1])-1)
        level.append(df.loc[df['ID'] == node_name, 'level'].values[0])

        if node_name == 'Elves':
            border_colors[-1] = 'red'
            # node_value_num[-1]=5

        # Hover text
        node_text.append(f'{node_name} ({node_value[-1]})<br>Level {level[-1]}')

    # Node size
    node_size = [round((adj+1)**1.5) for adj in node_value_num]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=node_size,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='node_size',
                xanchor='left',
                titleside='right'
            ),
            line=dict(
                color=border_colors,
                width=2
            )))

    node_trace.text = node_text

    # Create level edge trace
    level_x = []
    level_y = []
    for i in range(1, max(df['level'])):
      level_nodes = [node for node in G.nodes() if df.loc[df['ID'] == node, 'level'].values[0] == i]
      level_coord = np.array([pos[node] for node in level_nodes])
      alpha_x, alpha_y = alpha_shape(level_coord)
      for j in range(len(alpha_x)):
        if j==len(alpha_x)-1:
          level_x.extend([alpha_x[j], alpha_x[0], None])
          level_y.extend([alpha_y[j], alpha_y[0], None])
        else:
          level_x.extend([alpha_x[j], alpha_x[j+1], None])
          level_y.extend([alpha_y[j], alpha_y[j+1], None])

    level_edge_trace = go.Scatter(
        x=level_x, y=level_y,
        line=dict(width=0.5, color='red'),
        hoverinfo='none',
        mode='lines',
        opacity=0.4)

    # Run the Louvain community detection algorithm
    communities = community_louvain.best_partition(G, resolution=1)

    # Define a color palette for the communities
    num_communities = len(set(communities.values()))
    color_palette = px.colors.qualitative.Set1

    # Create rectangles for each community
    rect_list = []
    for community_id, color in zip(set(communities.values()), color_palette):
        community_nodes = [node for node in G.nodes() if communities[node] == community_id]
        x_values = [pos[node][0] for node in community_nodes]
        y_values = [pos[node][1] for node in community_nodes]

        bound_fix = max(node_size)/250
        x_range = [min(x_values)-bound_fix, max(x_values)+bound_fix]
        y_range = [min(y_values)-bound_fix, max(y_values)+bound_fix]

        rectangle = go.layout.Shape(
            type='rect',
            x0=x_range[0],
            x1=x_range[1],
            y0=y_range[0],
            y1=y_range[1],
            fillcolor=color,
            line=dict(color="black", width=0),
            opacity=0.2
        )
        rect_list.append(rectangle)

    # Update layout with rectangles
    layout = go.Layout(
        title='Network map - Elf',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[dict(
            text="Generated on " + datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')+
                " | <a href='https://tmt1611.github.io/sunburst'> Sunburst</a>",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002)],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        shapes=[],
    )

    # Add the button to the layout
    layout['updatemenus'] = [dict(type='buttons', showactive=True,
                                  buttons=[dict(label='Estimate community OFF', method='relayout',
                                              args=[{'shapes': []}]),
                                           dict(label='Estimate community ON', method='relayout',
                                              args=[{'shapes': rect_list}]),
                                           dict(label="Toggle level line OFF", method="restyle",
                                                args=["visible", [True, True, False]]),
                                           dict(label="Toggle level line ON", method="restyle",
                                                args=["visible", [True, True, True]],),])]


    # Create the figure and display the graph
    fig = go.Figure(data=[edge_trace, node_trace, level_edge_trace], layout=layout)

    return fig, edge_trace, node_trace, level_edge_trace


def random_pos(G, n_nodes):
    pos = {}
    for i in range(n_nodes):
        pos[i] = (random.uniform(0, 1), random.uniform(0, 1))
    return pos

def generate_pos(G):
    pos = nx.spring_layout(G, pos=random_pos(G, len(G)), iterations=1000, k=.7)
    # pos = nx.circular_layout(G)
    # pos = nx.shell_layout(G)
    # pos = nx.spectral_layout(G)
    # pos = nx.kamada_kawai_layout(G)
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]
    return pos

if __name__ == "__main__":
    # Replace 'your_link_to_csv' with the actual URL of the CSV file
    url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTiwummYiOCTzagxsbho_ZAFxTpluojPuF6Ynaxjtato5r965ppSx0ST_XPQuwzxSpP_BcF51VMuprM/pub?output=xlsx'
    
    # Load the Excel file into a DataFrame
    df = pd.read_excel(url)
    df = df.fillna(method='ffill', limit=1).fillna('')
    df['parent'] = df['parent'].str.replace('<br>', ' ')
    df['ID'] = df['ID'].str.replace('<br>', ' ')

    fig4, edge_trace, node_trace, level_edge_trace = create_network_graph(df)
    fig4.write_html("index.html", include_plotlyjs="cdn")
