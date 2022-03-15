import re
from bs4 import BeautifulSoup
from langdetect import detect
from tqdm.auto import tqdm
import logging
import os
import json

dirname = os.path.dirname(__file__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _text_cleaner(text):
    """Text cleaning function. Removes HTML tags, escaped characters (e.g. \n)
    and removes NewsAPI text patterns and URLs

    Args:
        text (string): text of a news article

    Returns:
        string: cleaned text
    """
    # Removes HTML tags
    text = BeautifulSoup(text, features="lxml").get_text()
    # Remove escaped characters
    escapes = ''.join([chr(char) for char in range(1, 32)])
    text = text.translate(str.maketrans('', '', escapes))
    # Remove patterns
    expressions = ['… [+ [0-9]*chars]$', '…', 'https?://\S+']
    for i in expressions:
        text = re.sub(i, '', text)
    return text


def _detect_non_english(text):
    """Function that detects if there's non-english characters in text

    Args:
        text (string): text of a news article

    Returns:
        boolean: True if there's non-english characters exist
    """
    # korean
    if re.search("[\uac00-\ud7a3]", text):
        return True
    # japanese
    if re.search("[\u3040-\u30ff]", text):
        return True
    # chinese
    if re.search("[\u4e00-\u9FFF]", text):
        return True
    # arabic
    if re.search("[\u0600-\u06FF]", text):
        return True
    # devanagari (hindi)
    if re.search("[\u0900-\u097F]", text):
        return True
    return False


def string_cleaner(doc):
    """Cleans the text by applying _detect_non_english and _text_cleaner

    Args:
        doc (dict): document as a dictionary

    Returns:
        tuple: string: cleaned text, bool: whether to include the document
    """
    concat_text = " ".join(
        map(lambda x: '' if x is None else x, [doc['title'], doc['description'], doc['content']])
    )
    # Check if concatenated text is in English or not
    if detect(concat_text) != 'en' or _detect_non_english(concat_text):
        return None
    else:
        # Clean the text in 'title', 'description' and 'content' fields
        for key in ['title', 'description', 'content']:
            if doc[key]:
                doc[key] = _text_cleaner(doc[key])
        # Concatenating the fields with #SEPTAG# to separate them in the future
        doc['concat_text'] = "#SEPTAG#".join(
            map(lambda x: '' if x is None else x, [doc['title'], doc['description'], doc['content']])
        )
        return doc


def documents_cleaner(documents):
    # Cleaning the documents
    logger.info("Cleaning the documents.")
    dicts=[]
    with tqdm(total=len(documents), position=0, unit="Docs", desc="Cleaning documents") as progress_bar:
        for doc in documents:
            # TODO: make the for loop faster (parallelization?)
            if (doc['description']=='' or doc['description']==None) and (doc['content']=='' or doc['content']==None):
                # Don't include documents without content and description
                continue
            doc = string_cleaner(doc)
            if doc:
                dicts.append(
                    {
                        'text': doc['concat_text'], 
                        'meta': {
                            'source': doc['source'],
                            'publishedat': doc['publishedAt'],
                            'url': doc['url'],
                            'urltoimage': doc['urlToImage'],
                            'category': doc['category']
                        }
                    }
                )
            progress_bar.update()
    
    # Go over results_list and remove duplicates based on 'text'f
    logger.info("Removing the duplicated documents.")
    holder = {}
    for d in dicts:
        holder.setdefault(d['text'], d['meta'])  # each key is unique
    # Reformat into list of dictionaries
    dicts = [{'text': k, 'meta': v} for k, v in holder.items()]

    return dicts


def clean_backups(backups_dir):
    # Check there is documents to load
    backup_files = ["mongodb_top_headlines.json", "mongodb_everything.json"]
    
    if not all([os.path.isfile(os.path.join(backups_dir, i)) for i in backup_files]):
        logger.info("No backup files to load into the document store.")
    else:
        # Load each JSON in backups and add everything to documents
        logger.info(f"Loading the raw backups: {backup_files}.")
        documents = []
        for bak in backup_files:
            with open(os.path.join(backups_dir, bak), 'r') as file:
                for line in file:
                    documents.append(json.loads(line))
        
        # Cleaning the documents
        dicts = documents_cleaner(documents)

        # Exporting the cleaned documents to a json
        logger.info("Exporting the cleaned documents.")
        with open(os.path.join(backups_dir, 'mongodb_cleaned_docs.json'), 'w') as f:
            f.write(json.dumps(dicts))


if __name__ == "__main__":
    backups_dir = os.path.join(dirname, '../../artifacts/backups')
    clean_backups(backups_dir)
