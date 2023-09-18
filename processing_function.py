from datasets import load_dataset
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import json
from processing_function import *


#Create list of dictionaries that contains the sentences and the labels.
def process_dataset(path):
    dataset = []
    sentences = []
    labels = []
    with open(path,"r") as file:
        for line in file :

            tokens = line.strip().split()
            if len(tokens) != 0 : 
                sentences.append(tokens[0])
                labels.append(tokens[3])
            else :
                dataset.append({'sentences':sentences, 'labels': labels})
                sentences = []
                labels = []
    return dataset



def encode_label(dataset):
    #Encode labels
    full_labels=[]

    for l in dataset:
        full_labels.extend(l['labels'])
        unique_labels = list(set(full_labels))
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(unique_labels)
    dataset['encoded_label'] = label_encoder.transform(dataset['labels'])
    return dataset


def tokenize(dataset):
#for idx,  d in enumerate(dataset):
        tokenized_inputs = tokenizer(dataset['sentences'],is_split_into_words=True)
        return tokenized_inputs

def words_ids(dataset):
    dataset['word_ids'] = dataset['inputs'].word_ids()


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label < 4 :
                label += 4
            new_labels.append(label)

    return new_labels

def align(dataset):
    dataset['new_label'] = align_labels_with_tokens(dataset['encoded_label'],dataset['inputs'].word_ids())
    return dataset