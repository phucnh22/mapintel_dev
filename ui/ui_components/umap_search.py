from textwrap import wrap

from pandas import DataFrame

from ui.vis_components.umap import umap_plot


def umap_page(documents: DataFrame, query: dict, unique_topics: list):
    # Set custom data
    custom_data = documents["answer"].str.split("#SEPTAG#", expand=True, n=1)
    custom_data = custom_data.applymap(
        lambda t: "<br>".join(wrap(t.replace("#SEPTAG#", " "), width=100)) if t else ""
    )

    # Set query row
    if query:
        query_label = query["query_text"]
        if len(query_label) > 40:
            query_label = query_label[:40] + "..."
        documents = documents.append(
            {
                "answer": query["query_text"],
                "umap_embeddings_x": query["query_umap"][0],
                "umap_embeddings_y": query["query_umap"][1],
                "topic": query_label,
            },
            ignore_index=True,
        )
    else:
        query_label = "Query"

    # Produce umap plot and get configurations
    p, config = umap_plot(
        documents=documents,
        unique_topics=unique_topics,
        query_label=query_label,
        custom_data=custom_data,
    )

    return p, config
