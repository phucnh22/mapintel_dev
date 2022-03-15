import plotly.graph_objects as go
from ui.vis_components.utils import cat_to_color


def umap_plot(documents, unique_topics, query_label, custom_data):
    # Initialize the figure
    fig = go.Figure(
        layout=dict(
            height=850,
            width=1500,
            legend=dict(
                title_text="Topics (select legend entries to hide points)\n",
                title_font_size=13,
                itemsizing='constant'
            )
        )
    )
    
    # Add traces for each topic
    for c, topic in zip(cat_to_color(unique_topics), unique_topics):
        # Get data for the selected topic
        ix_mask = documents['topic'] == topic
        data = documents.loc[ix_mask]
        customd = custom_data.loc[ix_mask]

        # Add topics traces
        fig.add_trace(
            go.Scattergl(
                mode='markers',
                x=data['umap_embeddings_x'], 
                y=data['umap_embeddings_y'],
                text=data['topic'],
                customdata=customd,
                opacity=0.5,  # trace opacity
                marker=dict(
                    size=4,
                    color=c,
                    opacity=0.5,  # marker opacity
                    symbol='circle'
                ),
                name=topic,
                hovertemplate = "<b>%{customdata[0]}</b><br><br>" + \
                                "<b>Topic</b>: %{text}<br>" + \
                                "<b>Content</b>: %{customdata[1]}" + \
                                "<extra></extra>"
            )
        )

    # Add query trace (last so the marker is in front)
    ix_mask = documents['topic'] == query_label
    data = documents.loc[ix_mask]
    customd = custom_data.loc[ix_mask]
    fig.add_trace(
        go.Scattergl(
            mode='markers',
            x=data['umap_embeddings_x'], 
            y=data['umap_embeddings_y'],
            text=data['topic'],
            opacity=0.5,  # trace opacity
            marker=dict(
                size=20,
                color=cat_to_color([query_label]),
                opacity=1,  # marker opacity
                symbol='x-dot',
                line=dict(
                    color='black',
                    width=1
                )
            ),
            name=query_label,
            hovertemplate = "<b>Query Marker</b><br><br>" + \
                            "<b>Query</b>: %{text}<br>" + \
                            "<extra></extra>"
        )
    )

    # Define plot configurations options
    config = {
        'scrollZoom': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': [
            'select2d', 
            'lasso2d', 
            'autoScale2d', 
            'toggleSpikelines', 
            'hoverCompareCartesian'
        ]
    }

    return fig, config
