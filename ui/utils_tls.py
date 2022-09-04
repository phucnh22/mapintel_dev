import os
import argparse
import pathlib
import spacy
import arrow
import subprocess
import collections
import shutil
import datetime
import codecs
from xml.etree import ElementTree
from news_tls.data import Token, Sentence, Article
import plotly.graph_objects as go
import numpy as np
import argparse
from pathlib import Path
from news_tls import utils, datewise, clust, summarizers
from pprint import pprint
from news_tls.data import ArticleCollection

#------------------------------------------------#
######------ TOKENIZE DATA ------######
#------------------------------------------------#


def tokenize_dataset(articles, spacy_model):
    nlp = spacy.load(spacy_model)        
    out_batch = []
    for i, a in enumerate(articles):
        tokenized_doc = ''
        a['text']=a['text'].replace("#SEPTAG#"," ")
        doc = nlp(a['text'])
        for sent in doc.sents:
            tokens = [tok.text for tok in sent if not tok.text.isspace()]
            tokenized_doc += ' '.join(tokens) + '\n'
        a['text'] = tokenized_doc.strip()
        out_batch.append(a)
        if i % 50 == 0:
            print(f'Processed {i} articles')
    return out_batch

#------------------------------------------------#
######------ HEIDELTIME ------######
#------------------------------------------------#

def write_input_articles(articles, out_dir):
    utils.force_mkdir(out_dir)
    date_to_articles = collections.defaultdict(list)
    for a in articles:
        date = arrow.get(a['time']).datetime.date()
        date_to_articles[date].append(a)

    for date in sorted(date_to_articles):
        utils.force_mkdir(out_dir / str(date))
        date_articles = date_to_articles[date]
        for a in date_articles:
            fpath = out_dir / str(date) / '{}.txt'.format(a['id'])
            with open(fpath, 'w') as f:
                f.write(a['text'])


def delete_input_articles(articles, out_dir):
    date_to_articles = collections.defaultdict(list)
    for a in articles:
        date = arrow.get(a['time']).datetime.date()
        date_to_articles[date].append(a)

    for date in sorted(date_to_articles):
        date_articles = date_to_articles[date]
        for a in date_articles:
            fpath = out_dir / str(date) / '{}.txt'.format(a['id'])
            if os.path.exists(fpath):
                os.remove(fpath)


def heideltime_preprocess(articles, dataset_dir, heideltime_path):
    apply_heideltime = heideltime_path / 'apply-heideltime.jar'
    heideltime_config = heideltime_path / 'config.props'
    
    out_dir = dataset_dir  / 'time_annotated'
    utils.force_mkdir(out_dir)
    write_input_articles(articles, out_dir)
    subprocess.run([
        'java',
        '-jar',
        str(apply_heideltime),
        str(heideltime_config),
        str(out_dir),
        'txt'
    ],check=True)
    delete_input_articles(articles, out_dir)

def heideltime_main(data, dataset_dir, heideltime):
    dataset_dir = pathlib.Path(dataset_dir)
    heideltime_path = pathlib.Path(heideltime)
    if not dataset_dir.exists():
        raise FileNotFoundError('dataset not found')
    if not heideltime_path.exists():
        raise FileNotFoundError('heideltime not found')
    heideltime_preprocess(data, dataset_dir, heideltime_path)

############### HEIDEILTIME END ##################


#------------------------------------------------#
######------ PREPROCESSING WITH SPACY ------######
#------------------------------------------------#

def extract_time_tag_value(time_tag):
    value = [(None, None)]

    if 'type' not in time_tag.attrib:
        return value
    elif time_tag.attrib['type'] == 'DATE':
        formats = ['%Y-%m-%d', '%Y-%m', '%Y']
    elif time_tag.attrib['type'] == 'TIME':
        formats = ['%Y-%m-%dT%H:%M', '%Y-%m-%dTMO', '%Y-%m-%dTEV',
                   '%Y-%m-%dTNI', '%Y-%m-%dTAF']
    else:
        return value

    for format in formats:
        try:
            time = datetime.datetime.strptime(
                time_tag.attrib['value'], format)
            value = [(time, format)]
        except:
            pass
    return value


def parse_timeml_doc(raw):

    # cleanup heideltime bugs
    replace_pairs = [
        ("T24", "T12"),
        (")TMO", "TMO"),
        (")TAF", "TAF"),
        (")TEV", "TEV"),
        (")TNI", "TNI"),
    ]
    for old, new in replace_pairs:
        raw = raw.replace(old, new)

    tokens = []
    time_values = []

    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as e:
        return None, None

    tokens.extend(root.text.split())
    time_values.extend([(None, None)] * len(tokens))

    for time_tag in root:
        if time_tag.text is None:
            continue
        split_text = time_tag.text.split()
        tokens.extend(split_text)
        value = extract_time_tag_value(time_tag)
        time_values.extend(value * len(split_text))
        split_tail = time_tag.tail.split()
        tokens.extend(split_tail)
        time_values.extend([(None, None)] * len(split_tail))

    return tokens, time_values


def read_articles(articles, tmp_dir):
    date_to_articles = collections.defaultdict(list)
    for a in articles:
        date = arrow.get(a['time']).date()
        date_to_articles[date].append(a)
    for date in sorted(date_to_articles):
        date_articles = date_to_articles[date]
        for a in date_articles:
            fpath = tmp_dir / str(date) / '{}.txt.timeml'.format(a['id'])
            if os.path.exists(fpath):
                with codecs.open(fpath, 'r', encoding='utf-8') as f:
                    raw = f.read()
                yield a, raw


def preprocess_title(title, pub_time, nlp):
    doc = nlp(title)
    token_objects = []
    for token in doc:
        token_object = Token(
            token.orth_,
            token.lemma_,
            token.tag_,
            token.ent_type_,
            token.ent_iob_,
            token.dep_,
            token.head.i,
            None,
            None,
        )
        token_objects.append(token_object)
    title_object = Sentence(title, token_objects, pub_time, None, None)
    return title_object


def preprocess_article(old_article, timeml_raw, nlp):
    tokens, time_values = parse_timeml_doc(timeml_raw)

    if tokens is None:
        return None

    doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
    nlp.tagger(doc)
    nlp.entity(doc)
    nlp.parser(doc)

    token_objects = []
    for token in doc:
        token_object = Token(
            token.orth_,
            token.lemma_,
            token.tag_,
            token.ent_type_,
            token.ent_iob_,
            token.dep_,
            token.head.i,
            time_values[token.i][0],
            time_values[token.i][1],
        )
        token_objects.append(token_object)

    sentence_objects = []
    for sent in doc.sents:
        sent_tokens = token_objects[sent.start:sent.end]
        times = [tok.time for tok in sent_tokens if tok.time]
        if times:
            time = times[0]
        else:
            time = None

        pub_time = arrow.get(old_article['time'])
        sent_object = Sentence(str(sent), sent_tokens, pub_time, time, None)
        sentence_objects.append(sent_object)

    raw_title = old_article.get('title')
    if raw_title:
        title_object = preprocess_title(raw_title, pub_time, nlp)
    else:
        title_object = None

    new_article = Article(
        title=raw_title,
        text=old_article['text'],
        time=old_article['time'],
        id=old_article.get('id'),
        sentences=sentence_objects,
        title_sentence=title_object
    )
    return new_article


def preprocess_dataset(articles, dataset_dir, nlp):
    h_output_dir = dataset_dir / 'time_annotated'
    out_path= dataset_dir / 'articles.preprocessed.jsonl'
    i = 0
    out_batch=[]
    for old_a, timeml_raw in read_articles(articles, h_output_dir):
        a = preprocess_article(old_a, timeml_raw, nlp)

        if a:
            out_batch.append(a.to_dict())
        else:
            date = arrow.get(old_a['time']).date()
            print('cannot process:', date, old_a['id'])

        if i % 50 == 0:
            print('processing batches,', i, 'articles done')
        i += 1
    utils.write_jsonl(out_batch, out_path, override=True)
    #gz_path = str(out_path) + '.gz'
    #utils.gzip_file(inpath=out_path, outpath=gz_path, delete_old=True)


def spacy_main(data, dataset_dir, spacy_model):
    dataset_dir = pathlib.Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError('dataset not found')
    nlp = spacy.load(spacy_model)
    preprocess_dataset(data, dataset_dir, nlp)
    print('Finished preprocessing!')

############### SPACY END ##################

def preprocessing_tls(data, dataset_dir, heideltime_dir , spacy_model='en_core_web_sm'):
    processing_data=tokenize_dataset(
        articles=data,
        spacy_model=spacy_model)
    heideltime_main(
        data = processing_data, 
        dataset_dir = dataset_dir, 
        heideltime = heideltime_dir)
    spacy_main(
        data = processing_data,
        dataset_dir = dataset_dir,
        spacy_model = spacy_model)

#------------------------------------------------#
######------ TLS ------######
#------------------------------------------------#

def run(tls_model, collection, outpath):
    outputs = []
    times = [a.time for a in collection.articles()]
    # setting start, end, L, K manually instead of from ground-truth
    collection.start = min(times)
    collection.end = max(times)
    l = 8 # timeline length (dates)
    k = 1 # number of sentences in each summary
    timeline = tls_model.predict(
        collection,
        max_dates=l,
        max_summary_sents=k,
    )

    print('*** TIMELINE ***')
    utils.print_tl(timeline)  
    outputs.append(timeline.to_dict())
    if outpath:
        utils.write_json(outputs, outpath)
    return timeline

def main_tls(dataset_dir, method, model = None, output = None):
    collection = ArticleCollection(dataset_dir)
    if method == 'datewise':
        # load regression models for date ranking
        key_to_model = utils.load_pkl(model)
        models = list(key_to_model.values())
        date_ranker = datewise.SupervisedDateRanker(method='regression')
        # there are multiple models (for cross-validation),
        # we just an arbitrary model, the first one
        date_ranker.model = models[0]
        sent_collector = datewise.PM_Mean_SentenceCollector(
            clip_sents=2, pub_end=2)
        summarizer = summarizers.CentroidOpt()
        system = datewise.DatewiseTimelineGenerator(
            date_ranker=date_ranker,
            summarizer=summarizer,
            sent_collector=sent_collector,
            key_to_model = key_to_model
        )
    elif method == 'clust':
        cluster_ranker = clust.ClusterDateMentionCountRanker()
        clusterer = clust.TemporalMarkovClusterer()
        summarizer = summarizers.CentroidOpt()
        system = clust.ClusteringTimelineGenerator(
            cluster_ranker=cluster_ranker,
            clusterer=clusterer,
            summarizer=summarizer,
            clip_sents=2,
            unique_dates=True,
        )
    else:
        raise ValueError(f'Method not found: {method}')
    return run(system, collection, output)


def timeline_plot(dates, labels):
    fig = go.Figure(
    layout=dict(
        autosize=True,
        clickmode="none",
        uniformtext=dict(minsize=25),
        margin = dict(t=1, l=1, r=1, b=1),
        height = 500,
        width= 800,
        legend=dict(orientation="h"),
        xaxis_range=[-10,10],
        paper_bgcolor='white',
        plot_bgcolor='white'            
        )
    )
    # Adding main line
    fig.add_trace(
        go.Scatter(
            x=np.zeros(len(dates)),
            y=dates,
            mode='lines',
            name='timeline'
            )
    )
    # Adding points to timeline
    fig.add_trace(
        go.Scatter(
            x=np.zeros(len(dates)),
            y=dates,
            mode='markers',
            marker=dict(
                color='red',
                size=9,
                line=dict(width=2,color='purple')),
            text=dates,
            #hoverinfo="text + value",
            hovertemplate="%{text}",
            name='events'
            )
    )
    # Adding text
    label_offsets = np.zeros(len(dates))
    label_offsets[::2] = 4
    label_offsets[1::2] = -4
    for i, (l, d) in enumerate(zip(labels, dates)):
        fig.add_annotation(
                x=label_offsets[i],
                y=d,
                text=l,
                showarrow=True,
                font=dict(
                    family="Ariel",
                    size=12,
                    color="#000000"
                    ),
                align="center",
                ax=label_offsets[i],
                ay=0,
                bordercolor="#c7c7c7",
                borderwidth=0,
                borderpad=2,
                bgcolor="#ff7f0e",
                opacity=0.8
                )
    return fig