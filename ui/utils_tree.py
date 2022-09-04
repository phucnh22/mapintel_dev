
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import textwrap
from newsapi import NewsApiClient
from googleNews import GoogleNews
from pandas import json_normalize
from time import mktime

@st.cache
def fetch_data_newsapi(api_key, categories):
    # genrate NewsApiClient
    newsapi = NewsApiClient(api_key=api_key)
    cols = ['source', 'author', 'title', 'description', 'url', 'urlToImage', 'publishedAt', 'content', 'category']
    final_data_frame = pd.DataFrame(columns=cols)
    for category in categories:
        top_headlines = newsapi.get_top_headlines(q=None,
                                            category=category,
                                            language='en',
                                            country='us')
        data_frame = pd.DataFrame(top_headlines['articles'])
        # add category to title
        data_frame['title'] = data_frame['title'] + ' | ' + category
        data_frame['category'] = category
        final_data_frame =final_data_frame.append(data_frame)
    return final_data_frame

@st.cache
def fetch_data_ggnews(topics = ['sports', 'health', 'technology', 'science', 'entertainment','business', 'world']):
    news = GoogleNews()
    cols = ['title', 'link', 'published', 'publishedAt', 'category']
    final_data_frame = pd.DataFrame(columns=cols)
    for topic in topics:
        result = news.topic_headlines(topic)
        df = json_normalize(result['entries'])
        df = df[['title','link','published','published_parsed']]
        df['title'] = df['title'] + ' | ' + topic
        df['category'] = topic
        df['published_parsed']=df['published_parsed'].map(mktime)
        df.rename(columns={'published_parsed':'publishedAt'},inplace=True)
        df.rename(columns={'link':'url'},inplace=True)
        final_data_frame =final_data_frame.append(df)
    return final_data_frame

def custom_wrap(s, width=15):
    """Insert line break for long text"""
    return "<br>".join(textwrap.wrap(s,width=width))

class Newsmap():
    def __init__(self, input_data, date_col = 'publishedAt', filter_col = 'category', value_col = 'treeRank', num_articles=10):
        """
        1. levels: hierarchy level of the treemap data
        2. date_column: name of column that specifies datetime in json format
        3. target_column: name of column that is being categorized
        4. value_column: name of column that contains numeric value to define box size
        5. len_article: number of article showed per box
        6. filter_list: list of selected categories to present in the map
        """       
        self.df = input_data
        self.date_col = date_col
        self.filter_col = filter_col
        self.value_col = value_col
        self.num_articles = num_articles

    def __custom_wrap(self, s, width=20):
        """Insert line break for long text"""
        return "<br>".join(textwrap.wrap(s,width=width))
    def __cut_string(self, s):
        """Cut string if its len is longer than 50 chars"""
        if len(s)>50:
            new_s = s[:50]
            new_s += "..."
        else: new_s = s
        return new_s
    def __filter_out_data(self, df, column, filter_list):
        """Filter data out of specific column within a dataframe"""
        return df.loc[df[column].isin(filter_list),:]
    def __rank_data_generator(self, num_sequence):
        """Generate a slowly decreasing sequence for the box size in treemap"""
        seq=[]
        n = 120
        for i in range(num_sequence):
            if i<2:
                seq.append(n/2)
            elif i<5:
                seq.append(n/3)
            elif i<9:
                seq.append(n/4)
            else:
                seq.append(n/5)
        return seq
    def __trim_data(self, newsapi=None):
        """To convert json datetime format into shorter version of number for easily comparing
        Then cut of the top most recent articles"""
        # if using NewsAPI , need to remove all special characters, trim the number to reduce size, convert to interger
        if newsapi:
            self.df[self.value_col] = self.df[self.date_col].str.replace('[^0-9]+', '').str.slice(start=2, stop=12, step=None).astype(int)
        # assign value as date if using normal tree rank by published date
        if self.value_col=="treeRank":
            self.df[self.value_col] = self.df[self.date_col]   
        categories=self.df[self.filter_col].unique()
        trim_df = pd.DataFrame(columns=self.df.columns)
        for i in categories:
            temp_df = self.df[self.df[self.filter_col]==i].sort_values(self.value_col, ascending=False).head(self.num_articles)  
            if self.value_col=="treeRank":
                temp_df[self.value_col] = self.__rank_data_generator(num_sequence=len(temp_df))    #assign the list as value
                trim_df = trim_df.append(temp_df)
            elif self.value_col=="relevance":
                trim_df = trim_df.append(temp_df)
            else:
                raise Exception('Value to rank tree data not found')             
        return trim_df

    def __build_hierarchy_tree_data(self, data, levels = ['title','category'], add_cols = ['hover_text','url']):
        """Build dataframe to fit input for treemap"""
        if len(add_cols) >= 1:
            df_all_trees = pd.DataFrame(columns=['id', 'parent', 'value'] + add_cols)
            for i, level in enumerate(levels):
                df_tree = pd.DataFrame(columns=['id', 'parent', 'value'] + add_cols)
                # assign id values
                if i == 0: # Effort to generate full text in hover  
                    dfg = data.groupby(levels[i:]+add_cols).sum().reset_index()
                    df_tree['id'] = dfg[level].copy()
                    df_tree[add_cols] = dfg[add_cols].copy()
                else:
                    dfg = data.groupby(levels[i:]).sum().reset_index()
                    df_tree['id'] = dfg[level].copy()
                    for column in add_cols:
                        df_tree[column] = df_tree['id']
                # assign parent values
                if i < len(levels) - 1:
                    df_tree['parent'] = dfg[levels[i+1]].copy()
                else:
                    df_tree['parent'] = 'total'
                # assign value values
                df_tree['value'] = dfg[self.value_col]
                # all trees df
                df_all_trees = df_all_trees.append(df_tree, ignore_index=True)
            total = pd.Series(dict(id='total', parent='',
                                    value=data[self.value_col].sum()))
            df_all_trees = df_all_trees.append(total, ignore_index=True)     
        return df_all_trees

    def pre_processing(self, filter_list):
        """Pre-processing input data to generate dataframe for Plotly Treemap
        """
        df_trees = self.__trim_data()
        # wrap break line for title
        df_trees['title'] = df_trees['title'].map(self.__custom_wrap)
        df_trees['hover_text'] = df_trees['title']
        # cut string if longer than 50 chars 
        #df_new['title'] = df_new['title'].map(self.__cut_string)
        self.df_trees = self.__build_hierarchy_tree_data(data=self.__filter_out_data(df_trees, self.filter_col, filter_list))   
    def tree_map(self):
        df_treemap = self.df_trees
        """Create a plotly Treemap object"""
        fig = go.Figure(
            layout=dict(
                autosize=True,
                clickmode="none",
                uniformtext=dict(minsize=25),
                margin = dict(t=1, l=1, r=1, b=1),
                height = 800,
                width= 800,
                treemapcolorway = ['rgba(68,156,31,255)', 'rgba(31,156,102,255)', 'rgba(28,94,141,255)', 'rgba(147,29,29,255)', 'rgba(152,30,136,255)', 'rgba(64,30,151,255)', 'rgba(121,106,24,255)'],
            ),
        )
        """Create a plotly Treemap object"""
        text_list=df_treemap.id
        fig.add_trace(
            go.Treemap(
                labels = df_treemap.hover_text,
                text =  text_list,
                parents = df_treemap.parent,
                values = df_treemap.value,
                branchvalues='total',
                textinfo ="text",
                #hoverinfo='skip',
                hovertemplate='<b>Score: %{value} </b> <br> %{label}<br><extra></extra>',
                root_color="lightgrey",
                textfont_size= 18,
                pathbar_textfont_size=1,
                maxdepth=3,
                outsidetextfont = {
                    'size':1},
                pathbar = {
                    'side':'bottom',
                    'thickness':12,
                    'visible':False},
                tiling={
                    'squarifyratio':1.7},
                marker=dict(line_width=0)
            ),  
        )
        config= {
        "displaylogo": False,
        }
        return fig, config