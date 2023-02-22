import os
import nltk
import torch
import numpy as np
from urllib.request import urlretrieve
from zipfile import ZipFile

from src.utils.pyutils import save_file, load_file


def tokenize_questions(data):
    for d in data:
        q = d['question'].lower()
        q = nltk.wordpunct_tokenize(q)
        d['tokens'] = q
    return data


def get_tokens_length_frequency(data):
    temp = {}

    for d in data:
        ql = len(d['tokens'])
        if ql not in temp:
            temp[ql] = 1
        else:
            temp[ql] += 1

    temp = sorted(temp.items(), key=lambda x: x[0])
    temp_dict = {x[0]: x[1] for x in temp}

    return temp_dict


def filter_by_tokens_length(data, tokens_length):

    for d in data:
        d['tokens'] = d['tokens'][:tokens_length]
    return data


def get_token_embeddings(save_path, save_name):

    # vector_size: 300
    # Window: 5
    # Corpus: English Wikipedia Dump of February 2017; Gigaword 5th Edition
    # vocabulary_size: 291392
    # Algorithm: fastText Skipgram
    # Lemmatization: False

    file_path = os.path.join(save_path, save_name)
    zip_path = os.path.join(save_path, '22.zip')

    if os.path.exists(file_path):
        embeddings = load_file(path=file_path)
    else:
        if not os.path.exists(zip_path):
            urlretrieve(url='http://vectors.nlpl.eu/repository/20/22.zip', filename=zip_path)
            print('Downloading Skipgram Embeddings')
        with ZipFile(file=zip_path, mode='r') as ref:
            ref.extractall(path=save_path)
        print('Extracting Skipgram Embeddings')

        with open(file=os.path.join(save_path, 'model.txt'), mode='r') as f:
            model = f.readlines()

        embeddings = {}

        for i in range(1, len(model)):
            temp = model[i].split(sep=' ')
            word = temp[0]
            vector = np.array([float(v) for v in temp[1:-1]])
            embeddings[word] = vector

        save_file(data=embeddings, path=save_path, file_name=save_name, file_type='pickle')

    return embeddings
