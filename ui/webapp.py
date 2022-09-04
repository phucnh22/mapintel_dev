import os
import sys
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, "../"))

from ui.ui_components.umap_search import umap_page
from ui.utils import (
    doc_count,
    feedback_doc,
    get_all_docs,
    retrieve_doc,
    topic_names,
    umap_query,
)
# Treemap components
from streamlit_plotly_events import plotly_events
from ui.utils_tree import (
    custom_wrap,
    Newsmap
)
# Timeline components
from ui.vis_components.timelines import (
    pre_processing_timeline,
    timeline_plot,
    date_filter,
    group_by_date
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
default_question = "Bitcoin Crypto "
unique_topics = topic_names()
debug = False
batch_size = 10000
filters = []
# treemap variables
articles_categories = ['sports', 'health', 'technology', 'science', 'entertainment','business', 'general']
news_api_key = '99ca995c8d9349848711d2942b0c0d72'

# Set page configuration
st.set_page_config(page_title="NewsIntel App", layout="wide")

# UI sidebar
with st.sidebar:
    st.header("Options:")
    with st.form(key="options_form"):
        end_of_week = date.today() + timedelta(6 - date.today().weekday())
        _, mid, _ = st.columns([1, 10, 1])
        with mid:  # Use columns to avoid slider labels being off-window
            filter_date = st.slider(
                "Date range",
                min_value=date(2020, 1, 1),
                value=(date(2020, 1, 1), end_of_week),
                step=timedelta(7),
                format="DD-MM-YY",
            )
        with st.expander("Query Options"):
            filter_category = st.multiselect(
                "Category", options=unique_topics, default=None
            )
            filter_category_exclude = st.checkbox("Exclude", value=True)
        with st.expander("Results Options"):
            top_k_reader = st.slider(
                "Number of returned documents",
                min_value=1,
                max_value=20,
                value=10,
                step=1,
            )
            top_k_retriever = st.slider(
                "Number of candidate documents",
                min_value=1,
                max_value=200,
                value=100,
                step=1,
            )
        with st.expander("Visualization Options"):
            umap_perc = st.slider(
                "Percentage of documents displayed",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                help="Display a randomly sampled percentage of the documents to improve performance",
            )
        with st.expander("Treemap Options"):
            a_cnt = st.slider(
                label="Number of articles",
                min_value=10,
                max_value=20,
                step=2,
                value=10
            )
            # Select categories to display in the treemap
            news_cat_options = st.multiselect(
                label="Categories",
                options=articles_categories,
                #default=['general', 'health', 'sports', 'business']
                default=articles_categories
            )       
        st.form_submit_button(label="Submit")

# Prepare filters
if filter_category:
    filter_topics = list(map(lambda x: x.lower(), filter_category))

    # If filters should be excluded
    if filter_category_exclude:
        filter_topics = list(set(unique_topics).difference(set(filter_topics)))

    # Sort filters
    filter_topics.sort(key=lambda x: int(x.split("_")[0]))

    filters.append({"terms": {"topic_label": filter_topics}})
else:
    filter_topics = unique_topics

filters.append(
    {
        "range": {
            "publishedat": {
                "gte": filter_date[0].strftime("%Y-%m-%d"),
                "lte": filter_date[1].strftime("%Y-%m-%d"),
            }
        }
    }
)

# Sampling the docs and passing them to the UMAP plot
doc_num = doc_count(filters)
# sample_size = int(umap_perc / 100 * doc_num) #normal size
sample_size = int(umap_perc / 500 * doc_num) # reduced size to increase performance

# Title
st.title("News Intel Application")
# Create a text element and let the reader know the data is loading.
#data_load_state = st.text('Loading data...')
#df = fetch_data(news_api_key, value)
#df = fetch_data_ggnews()
df = pd.read_csv("data/top_headlines.csv")
#data_load_state.text('Data loaded!')
cached_df = df.copy()


# LAYING OUT THE TOP SECTION OF THE APP
row1_1, row1_2 = st.columns((3, 2))
with row1_1:
    st.write(
        """
        ##
        Exploring the lastest news in the internet in different categories \n
        Or using free query to explore our news database
        """
    )
with row1_2:
    st.subheader("")
    ## Newsmap
    query_methods=['Lastest News', 'Free Query']
    query_method=st.radio(
        label='Select exploring method:',
        options=query_methods,
        )

# LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
row2_1, row2_2 = st.columns((3, 2))

with row2_1:
    st.subheader("Newsmap")
    # Search bar
    if query_method == 'Lastest News':
        newsmap = Newsmap(cached_df)
        newsmap.pre_processing(filter_list=news_cat_options)
        
        fig_tree, config_tree = newsmap.tree_map()
        #st.plotly_chart(fig_tree, use_container_width=True, config=config_tree)
        selected_points = plotly_events(
            fig_tree, 
            click_event=True, 
            hover_event=False,
            override_height=850,
            override_width='100%')
        try:
            # print plotly event
            selected_query = newsmap.df_trees.iloc[selected_points[0]["pointNumber"],3].split("-")[0].replace("<br>"," ")         
            link = newsmap.df_trees.iloc[selected_points[0]["pointNumber"],4]   
            st.write(f"{selected_query} - [Read online]({link})")
            #st.write(link)
        except:
            link=""
            st.write(selected_points)   
        # Question for API
        try:          
            question = selected_query
        except:
            question=""
        # Request to API
        results, raw_json = retrieve_doc(
            query=question,
            filters=filters,
            #top_k_reader=top_k_reader,
            top_k_reader=100, #for timeline developing purpose 
            top_k_retriever=top_k_retriever,
                )
        # timeline dataframe
        tl_df = pre_processing_timeline(results)
        

    if query_method == 'Free Query':
        question = st.text_input(label="Please provide your query:", value=default_question)
        # Request to API
        results, raw_json = retrieve_doc(
            query=question,
            filters=filters,
            #top_k_reader=top_k_reader,
            top_k_reader=100, #for timeline developing purpose 
            top_k_retriever=top_k_retriever,
        )
        # timeline dataframe
        tl_df = pre_processing_timeline(results)

        # Generate treemap
        newsmap = Newsmap(tl_df, date_col="date", value_col='relevance',num_articles=20)
        newsmap.pre_processing(filter_list=tl_df.category.unique())
        fig_tree, config_tree = newsmap.tree_map()
        
        selected_points = plotly_events(
            fig_tree, 
            click_event=True, 
            hover_event=False,
            override_height=850,
            override_width='100%')
        try:
            # print plotly event
            selected_query = newsmap.df_trees.iloc[selected_points[0]["pointNumber"],3].split("-")[0].replace("<br>"," ")         
            link = newsmap.df_trees.iloc[selected_points[0]["pointNumber"],4]   
            st.write(f"{selected_query} - [Read online]({link})")
        except:
            link=""
            st.write(selected_points)

    with st.expander("Expand/collapse the embedded article!:", expanded=False):
        if len(link) != 0:
            with st.spinner("Loading"):
                components.iframe(link, height=500, scrolling=True)

with row2_2:
    st.subheader("UMAP")
    st.write(f"Input query:\n {question}")
    with st.spinner(
        "Getting documents from database... \n " "Documents will be plotted when ready."
    ):
        # Read data for umap plot (create generator)
        umap_docs = get_all_docs(
            filters=filters, batch_size=batch_size, sample_size=sample_size
        )
    # Get results for query
    with st.spinner(
        "Performing neural search on documents... 🧠 \n "
        "Do you want to optimize speed or accuracy? \n"
        "Check out the docs: https://haystack.deepset.ai/docs/latest/optimizationmd "
    ):  
        # Plot the completed UMAP plot
        fig, config = umap_page(
            documents=pd.DataFrame(umap_docs),
            query=umap_query(question),
            unique_topics=filter_topics,
        )
        st.plotly_chart(fig, use_container_width=True, config=config)


st.subheader("Timeline from the Artiles related to the query")
row3_1, row3_2 = st.columns((5, 4))

try:
    with row3_1:
        timeline_fig=timeline_plot(tl_df)          
        st.plotly_chart(timeline_fig, use_container_width=True)
        # get the group of range for date filtering
        date_groups=group_by_date(tl_df.date.sort_values())  
        selected_date_range=st.select_slider(
            label="Select a range of dates",
            options=range(date_groups.shape[0]), 
            format_func=lambda x: date_groups.iloc[x,0],
            #value=date_groups[0],
            )
    with row3_2:
        printed_articles = date_filter(
                tl_df,
                from_date=date_groups.iloc[selected_date_range,1][0],
                to_date=date_groups.iloc[selected_date_range,1][1],
            ).reset_index()

        for index, row in printed_articles.sort_values('date',ascending=False).iterrows():
            st.markdown(
                 f"""
                 <p style='font-size: 15px;'>{row['date']} - 
                 <a href={row['url']}>{row['title']}</a>
                 </p>
                 """, 
                 unsafe_allow_html=True)
except:
    st.write("No input query")

# Make every button key unique
count = 0
raw_json_feedback = ""
print_result = False
if print_result:
    for result in results:
        # Define columns for answer text and image
        col1, _, col2 = st.beta_columns([6, 1, 3])
        with col1:
            title, description, content = result["answer"].split("#SEPTAG#")
            st.write(f"### {title}\n{description}\n\n{content}")
        with col2:
            if result["image_url"] is not None and result["image_url"] != "null":
                image_url = result["image_url"]
            else:
                image_url = "http://www.jennybeaumont.com/wp-content/uploads/2015/03/placeholder.gif"

            if result["url"] is not None:
                st.markdown(
                    f"""
                    <a href={result['url']}>
                        <img src={image_url} alt={result['url']} style="width:600px;"/>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <img src={image_url} alt="Placeholder Image" style="width:600px;">
                    """,
                    unsafe_allow_html=True,
                )

        "**Relevance:** ", result["relevance"], "**Topic:** ", result[
            "topic"
        ], "**Published At:** ", result["publishedat"][:-4].replace("T", ", ")

        # Define columns for feedback buttons
        button_col1, button_col2, _ = st.beta_columns([1, 1, 8])
        if button_col1.button(
            "👍", key=(result["answer"] + str(count)), help="Relevant document"
        ):
            raw_json_feedback = feedback_doc(
                question, result["answer"], result["document_id"], 1, "true", "true"
            )
            st.success("Thanks for your feedback")
        if button_col2.button(
            "👎", key=(result["answer"] + str(count)), help="Irrelevant document"
        ):
            raw_json_feedback = feedback_doc(
                question, result["answer"], result["document_id"], 1, "false", "false"
            )
            st.success("Thanks for your feedback!")
        count += 1
        st.write("___")
else:
    #st.write(f'Retreived total {len(results)} docs of {type(results)}')
    st.write("Phuc Nguyen - @NOVAIMS")
if debug:
    st.subheader("REST API JSON response")
    st.write(raw_json)
