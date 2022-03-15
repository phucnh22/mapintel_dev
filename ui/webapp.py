import os
import sys
from datetime import date, timedelta
import pandas as pd
import streamlit as st

dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, "../"))

from ui.ui_components.umap_search import umap_page
from ui.utils import (
    feedback_doc, 
    retrieve_doc, 
    get_all_docs,
    umap_query,
    topic_names,
    doc_count
)

# TODO: A problem with the application is that when setting the slider value the query
# will execute even if we didn't finish selecting the value we want. A temporary solution
# is to wrap the sliders inside a form component. The ideal solution would be to create
# a custom slider component that only updates the value when we release the mouse.
# TODO: When selecting a point in the UMAP, show the KNN points highlighted on the plot
# and the listed below (the action of selecting the point will call the query endpoint)
# TODO: Create a function to parse the query and allow search operators like "term" to
# restrict the query to documents that contain "term"

# Init variables
default_question = "Stock Market News"
unique_topics = topic_names()
debug = False
batch_size = 10000
filters = []

# Set page configuration
st.set_page_config(
    page_title = "MapIntel App",
    layout = "wide"
)

# Title
st.write("# Mapintel App")

# UI sidebar
with st.sidebar:
    st.header("Options:")
    with st.form(key="options_form"):
        end_of_week = date.today() + timedelta(6 - date.today().weekday())
        _, mid, _ = st.beta_columns([1, 10, 1])
        with mid:  # Use columns to avoid slider labels being off-window
            filter_date = st.slider(
                "Date range", 
                min_value=date(2020,1,1),
                value=(date(2020,1,1), end_of_week),
                step=timedelta(7), 
                format="DD-MM-YY"
            )
        with st.beta_expander("Query Options"):
            filter_category = st.multiselect(
                "Category",
                options=unique_topics,
                default="-1_news_covid_people_2021"
            )
            filter_category_exclude = st.checkbox(
                "Exclude",
                value=True
            )
        with st.beta_expander("Results Options"):
            top_k_reader = st.slider(
                "Number of returned documents",
                min_value=1, 
                max_value=20, 
                value=10, 
                step=1
            )
            top_k_retriever = st.slider(
                "Number of candidate documents", 
                min_value=1, 
                max_value=200, 
                value=100, 
                step=1
            )
        with st.beta_expander("Visualization Options"):
            umap_perc = st.slider(
                "Percentage of documents displayed", 
                min_value=1, 
                max_value=100, 
                value=10, 
                step=1, 
                help="Display a randomly sampled percentage of the documents to improve performance"
            )
        st.form_submit_button(label='Submit')

# Prepare filters
if filter_category:
    filter_topics = list(map(lambda x: x.lower(), filter_category))

    # If filters should be excluded
    if filter_category_exclude:
        filter_topics = list(set(unique_topics).difference(set(filter_topics)))

    # Sort filters
    filter_topics.sort(key=lambda x: int(x.split("_")[0]))

    filters.append(
        {
            "terms": {
                "topic_label": filter_topics
            }
        }
    )
else:
    filter_topics = unique_topics

filters.append(
    {
        "range": {
            "publishedat": {
                "gte": filter_date[0].strftime("%Y-%m-%d"),
                "lte": filter_date[1].strftime("%Y-%m-%d")
            }
        }
    }
)

# Search bar
question = st.text_input(label="Please provide your query:", value=default_question)

# TODO: create a umatrix endpoint much like the umap one and use it to display the umatrix?
# Sampling the docs and passing them to the UMAP plot
doc_num = doc_count(filters)
sample_size = int(umap_perc/100 * doc_num)
st.subheader("UMAP")
with st.spinner(
    "Getting documents from database... \n "
    "Documents will be plotted when ready."
):
    # Read data for umap plot (create generator)
    umap_docs = get_all_docs(
        filters=filters, 
        batch_size=batch_size, 
        sample_size=sample_size
    )

# Plot the completed UMAP plot
fig, config = umap_page(
    documents=pd.DataFrame(umap_docs), 
    query=umap_query(question),
    unique_topics=filter_topics
)
st.plotly_chart(fig, use_container_width=True, config=config)
st.write("___")

# Get results for query
with st.spinner(
    "Performing neural search on documents... 🧠 \n "
    "Do you want to optimize speed or accuracy? \n"
    "Check out the docs: https://haystack.deepset.ai/docs/latest/optimizationmd "
):
    results, raw_json = retrieve_doc(
        query=question, 
        filters=filters,
        top_k_reader=top_k_reader, 
        top_k_retriever=top_k_retriever
    )

st.write("## Retrieved answers:")

# Make every button key unique
count = 0
raw_json_feedback = ""

for result in results:
    # Define columns for answer text and image
    col1, _, col2 = st.beta_columns([6, 1, 3])
    with col1:
        title, description, content = result["answer"].split("#SEPTAG#")
        st.write(f"### {title}\n{description}\n\n{content}")
    with col2:
        if result['image_url'] is not None and result['image_url'] != "null":
            image_url = result['image_url']
        else:
            image_url = 'http://www.jennybeaumont.com/wp-content/uploads/2015/03/placeholder.gif'

        if result['url'] is not None:
            st.markdown(
                f"""
                <a href={result['url']}>
                    <img src={image_url} alt={result['url']} style="width:600px;"/>
                </a>
                """
                , unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <img src={image_url} alt="Placeholder Image" style="width:600px;">
                """
                , unsafe_allow_html=True
            )

    "**Relevance:** ", result["relevance"], "**Topic:** ", result["topic"], "**Published At:** ", result["publishedat"][:-4].replace('T', ', ')

    # Define columns for feedback buttons
    button_col1, button_col2, _ = st.beta_columns([1, 1, 8])
    if button_col1.button("👍", key=(result["answer"] + str(count)), help="Relevant document"):
        raw_json_feedback = feedback_doc(
            question, result["answer"], result["document_id"], 1, "true", "true"
        )
        st.success("Thanks for your feedback")
    if button_col2.button("👎", key=(result["answer"] + str(count)), help="Irrelevant document"):
        raw_json_feedback = feedback_doc(
            question, result["answer"], result["document_id"], 1, "false", "false"
        )
        st.success("Thanks for your feedback!")
    count += 1
    st.write("___")

if debug:
    st.subheader("REST API JSON response")
    st.write(raw_json)
