import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from vncorenlp import VnCoreNLP
import argparse
from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.layers import Input, Embedding, LSTM, Dropout, Dense, concatenate, Conv1D, MaxPooling1D, Flatten, Bidirectional, GlobalMaxPooling1D, Concatenate
from keras.models import Model
from keras.utils import plot_model

import warnings
warnings.filterwarnings("ignore")


# tạo bộ word segmentation cho tiếng Việt
rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

# load model và bpe lên theo hướng dẫn của PhoBERT
parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes',
    default="./PhoBERT_base_transformers/bpe.codes",
    required=False,
    type=str,
    help='path to fastBPE BPE'
)
args, unknown = parser.parse_known_args()
bpe = fastBPE(args)

# Load the dictionary
vocab = Dictionary()
vocab.add_from_file("./PhoBERT_base_transformers/dict.txt")

dir_path = os.path.dirname(os.path.abspath(__file__))

# Gọi file Bản sao của train.xlsx
path_train = os.path.join(dir_path, "Data_train.xlsx")
train_data = pd.read_excel(path_train)

# Gọi file Bản sao của test.xlsx
path_test = os.path.join(dir_path, "Data_test.xlsx")
test_data = pd.read_excel(path_test)


# Function to read an xlsx file
# Input: file path
# Output: DataFrame containing the file data
def read_xlsx(path):
    try:
        data = pd.read_excel(path, engine='openpyxl')
    except:
        data = pd.DataFrame()
    return data


# Function to extract the contents, titles, and ratings from the xlsx file
# Input: path to the folder (Train or Test)
# Output: the contents, titles, and ratings
def make_data(file_path):
    titles = []
    texts = []
    ratings = []

    data = read_xlsx(file_path)
    if data.empty:
        return titles, texts, ratings

    for index, row in data.iterrows():
        title = row['title']
        text = row['text']
        rating = row['rating']

        # Processing the content, title, and rating if needed
        # ...

        titles.append(title)
        texts.append(text)
        ratings.append(rating)

    return titles, texts, ratings

train_titles, train_texts, train_ratings = make_data(path_train)
test_titles, test_texts, test_ratings = make_data(path_test)

print(len(train_texts), len(train_titles), len(train_ratings))
print(len(test_texts), len(test_titles), len(test_ratings))

train_titles[0], train_texts[0], train_ratings[0]

# label encoded
lb_encoder = LabelEncoder()
lb_encoder.fit(train_ratings)

en_train_labels = lb_encoder.transform(train_ratings)
en_test_labels = lb_encoder.transform(test_ratings)

print(lb_encoder.classes_)  # in kiểm tra các labels

train_sent_titles, val_sent_titles, train_sent_texts, val_sent_texts, train_ratings, val_ratings = train_test_split(train_titles, train_texts, en_train_labels, test_size=0.1)
print(len(train_sent_texts), len(val_sent_texts))
print(len(train_sent_titles), len(val_sent_titles))

MAX_LEN = 256

def convert_sents_ids(sents):
    ids = []
    for sent in sents:
        sent = str(sent)
        subwords = '<s> ' + bpe.encode(sent) + ' </s>'
        encoded_sent = vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
        ids.append(encoded_sent)
    ids = pad_sequences(ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    return torch.tensor(ids)

train_title_ids = convert_sents_ids(train_sent_titles)
train_text_ids = convert_sents_ids(train_sent_texts)
val_title_ids = convert_sents_ids(val_sent_titles)
val_text_ids = convert_sents_ids(val_sent_texts)
test_title_ids = convert_sents_ids(test_titles)
test_text_ids = convert_sents_ids(test_texts)

def make_mask(batch_ids):
    batch_mask = []
    for ids in batch_ids:
        mask = [int(token_id > 0) for token_id in ids]
        batch_mask.append(mask)
    return torch.tensor(batch_mask)

train_title_masks = make_mask(train_title_ids)
train_text_masks = make_mask(train_text_ids)

val_title_masks = make_mask(val_title_ids)
val_text_masks = make_mask(val_text_ids)

test_title_masks = make_mask(test_title_ids)
test_text_masks = make_mask(test_text_ids)

def make_data_loader(ids, masks, BATCH_SIZE=4):
    data = TensorDataset(ids, masks)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=BATCH_SIZE)
    return dataloader

train_labels = torch.tensor(train_ratings)
val_labels = torch.tensor(val_ratings)
test_labels = torch.tensor(en_test_labels)

train_title_dataloader = make_data_loader(train_title_ids, train_title_masks)
train_text_dataloader = make_data_loader(train_text_ids, train_text_masks)

val_title_dataloader = make_data_loader(val_title_ids, val_title_masks)
val_text_dataloader = make_data_loader(val_text_ids, val_text_masks)

test_title_dataloader = make_data_loader(test_title_ids, test_title_masks)
test_title_dataloader = make_data_loader(test_text_ids, test_text_masks)

y_train = to_categorical(train_labels)
y_val = to_categorical(val_labels)

y_train_1d = np.argmax(y_train, axis=1)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

lr = LogisticRegression()
train_title_ids_2d = np.array(train_title_ids).reshape(len(train_title_ids), -1)
train_text_ids_2d = np.array(train_text_ids).reshape(len(train_text_ids), -1)

train_data = np.concatenate((train_title_ids_2d, train_text_ids_2d), axis=1)

lr.fit(train_data, y_train_1d)

test_title_ids_2d = np.array(test_title_ids).reshape(len(test_title_ids), -1)
test_text_ids_2d = np.array(test_text_ids).reshape(len(test_text_ids), -1)

test_data = np.concatenate((test_title_ids_2d, test_text_ids_2d), axis=1)

y_pred_lr = lr.predict(test_data)

acc_lr = accuracy_score(test_labels, y_pred_lr)
conf = confusion_matrix(test_labels, y_pred_lr)
clf_report = classification_report(test_labels, y_pred_lr)

print(f"Accuracy Score of Logistic Regression is: {acc_lr}")
print(f"Confusion Matrix:\n{conf}")
print(f"Classification Report:\n{clf_report}")
