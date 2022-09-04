import os
import sys
from datetime import date, timedelta, datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def pre_processing_timeline(results):  
    data=[dict(
        title=result["answer"].split("#SEPTAG#")[0].split(" - ")[0],
        url=result['url'],
        category=result['category'],
        relevance=result['relevance'],
        text=result["answer"].split("#SEPTAG#")[1], 
        date=datetime.strptime(result['publishedat'][:10], '%Y-%m-%d').date(),
        id=result['document_id']) 
                for result in results]
    df=pd.DataFrame(data)
    # Only return articles with relevance score >=0
    return df[df.relevance>=0]

def timeline_plot(df):
    plot_df = df.groupby('date').count()['id'].reset_index()
    fig = go.Figure(
                layout=dict(
                    autosize=True,
                    clickmode="none",
                    uniformtext=dict(minsize=25),
                    margin = dict(t=1, l=1, r=1, b=1),               
                    height = 200,
                    plot_bgcolor="white",
                    yaxis={'visible': False, 'showticklabels': False}
                ),
            )
    fig = fig.add_trace(
        go.Scatter(
            x=plot_df['date'], 
            y=plot_df['id'],
            mode='lines+markers',
            marker=dict(
                    color='lightblue',
                    size=9,
                    opacity=0.8,
                    line=dict(width=1,color='black')),
            line = dict(
                    color='lightblue',
                    dash="dot",
                    width=1.5
                    ),
            text=plot_df['id'],
            #hoverinfo="text + value",
            hovertemplate="<b> %{x} </b> <br> Number of articles: %{text}<extra></extra>"   
        )
    )
    return fig

def date_filter(data, from_date, to_date):
    if type(from_date) is str:
        from_date = datetime.strptime(from_date, '%Y-%m-%d').date()
    if type(to_date) is str:
        to_date = datetime.strptime(to_date, '%Y-%m-%d').date()
    return data[(data['date'] >= from_date) & (data['date'] < to_date)]

def group_by_date(date_list, step=15):
    groups=[]
    marks=date_list[::step].values
    for i in range(len(marks)):
        if i == (len(marks)-1):
            groups.append((marks[i].strftime(format='%Y-%m-%d'), date_list.max().strftime(format='%Y-%m-%d')))
        else:
            groups.append((marks[i].strftime(format='%Y-%m-%d'),marks[i+1].strftime(format='%Y-%m-%d')))
    return pd.DataFrame({'from':marks,'range':groups})

