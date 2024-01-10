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

# LSTM
EPOCHS = 100

vocab_size = len(vocab)
embedding_dim = 128
hidden_size = 256

# Đầu vào cho title
title_input = Input(shape=(train_title_ids.shape[1],))
title_embedding = Embedding(vocab_size, embedding_dim, input_length=train_title_ids.shape[1])(title_input)
title_lstm = LSTM(hidden_size, return_sequences=True)(title_embedding)
title_lstm_dropout = Dropout(0.2)(title_lstm)
title_lstm_final = LSTM(hidden_size)(title_lstm_dropout)

# Đầu vào cho text
text_input = Input(shape=(train_text_ids.shape[1],))
text_embedding = Embedding(vocab_size, embedding_dim, input_length=train_text_ids.shape[1])(text_input)
text_lstm = LSTM(hidden_size, return_sequences=True)(text_embedding)
text_lstm_dropout = Dropout(0.2)(text_lstm)
text_lstm_final = LSTM(hidden_size)(text_lstm_dropout)

# Kết hợp hai đầu vào
combined = concatenate([title_lstm_final, text_lstm_final])

# Các bước còn lại của mô hình
dense1 = Dense(128, activation='relu')(combined)
output = Dense(y_train.shape[1], activation='softmax')(dense1)

# Xây dựng mô hình
model_LSTM = Model(inputs=[title_input, text_input], outputs=output)

model_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_LSTM.summary()

model_LSTM_history = model_LSTM.fit(
    [np.array(train_title_ids), np.array(train_text_ids)],
    y_train,
    epochs=EPOCHS,
    batch_size=4,
    verbose=1,
    validation_data=([np.array(val_title_ids), np.array(val_text_ids)], y_val)
)

# Dự đoán trên tập validation
val_pred = model_LSTM.predict([np.array(val_title_ids), np.array(val_text_ids)])

# Chuyển đổi dự đoán về dạng categorical
val_pred_categorical = np.argmax(val_pred, axis=1)

#Tính các metrics
report = classification_report(np.argmax(y_val, axis=1), val_pred_categorical)
print(report)

model_LSTM.save_weights('LSTM_text_classification.h5')
# xây dựng hàm đánh giá
def test_LSTM(X_test, y_test):
    y_pred = model_LSTM.predict(X_test)
    pred = np.argmax(y_pred,axis=1)

    print(classification_report(y_test, pred))

# đánh giá trên tập test
test_LSTM([np.array(test_title_ids), np.array(test_text_ids)], np.array(test_labels))


# CNN
vocab_size = len(vocab)
embedding_dim = 128
num_filters = 128
filter_sizes = [3, 4, 5]

# Input for title
title_input = Input(shape=(train_title_ids.shape[1],))
title_embedding = Embedding(vocab_size, embedding_dim, input_length=train_title_ids.shape[1])(title_input)
title_conv_blocks = []
for filter_size in filter_sizes:
    title_conv = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(title_embedding)
    title_pool = MaxPooling1D(pool_size=train_title_ids.shape[1] - filter_size + 1)(title_conv)
    title_conv_blocks.append(title_pool)
title_concat = concatenate(title_conv_blocks, axis=-1)
title_flat = Flatten()(title_concat)

# Input for text
text_input = Input(shape=(train_text_ids.shape[1],))
text_embedding = Embedding(vocab_size, embedding_dim, input_length=train_text_ids.shape[1])(text_input)
text_conv_blocks = []
for filter_size in filter_sizes:
    text_conv = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(text_embedding)
    text_pool = MaxPooling1D(pool_size=train_text_ids.shape[1] - filter_size + 1)(text_conv)
    text_conv_blocks.append(text_pool)
text_concat = concatenate(text_conv_blocks, axis=-1)
text_flat = Flatten()(text_concat)

# Combine the two inputs
combined = concatenate([title_flat, text_flat])

# Additional layers of the model
dense1 = Dense(128, activation='relu')(combined)
output = Dense(y_train.shape[1], activation='softmax')(dense1)

# Build the model
model_CNN = Model(inputs=[title_input, text_input], outputs=output)

model_CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_CNN.summary()

def train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=4):
    history = model.fit(
        [X_train['title'], X_train['text']],
        y_train,
        validation_data=([X_val['title'], X_val['text']], y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    return history

history = model_CNN.fit(
    [np.array(train_title_ids), np.array(train_text_ids)],
    y_train,
    epochs=EPOCHS,
    batch_size=4,
    verbose=1,
    validation_data=([np.array(val_title_ids), np.array(val_text_ids)], y_val)
)

# Dự đoán trên tập validation
val_pred = model_CNN.predict([np.array(val_title_ids), np.array(val_text_ids)])

# Chuyển đổi dự đoán về dạng categorical
val_pred_categorical = np.argmax(val_pred, axis=1)

# Tính các metrics
report = classification_report(np.argmax(y_val, axis=1), val_pred_categorical)
print(report)

# xây dựng hàm đánh giá
def test_CNN(X_test, y_test):
    y_pred = model_CNN.predict(X_test)
    pred = np.argmax(y_pred,axis=1)

    print(classification_report(y_test, pred))

# đánh giá trên tập test
test_CNN([np.array(test_title_ids), np.array(test_text_ids)], np.array(test_labels))


# BiLSTM
def build_bilstm_model():
    # Define input layers for the title and text inputs
    title_input = Input(shape=(train_title_ids.shape[1],))
    text_input = Input(shape=(train_text_ids.shape[1],))

    # Embedding layer for title
    title_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, trainable=True)(title_input)
    # Embedding layer for text
    text_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, trainable=True)(text_input)

    # Bidirectional LSTM layer for title
    title_bilstm = Bidirectional(LSTM(64, return_sequences=True))(title_embedding)
    # Bidirectional LSTM layer for text
    text_bilstm = Bidirectional(LSTM(64, return_sequences=True))(text_embedding)

    # Global Max Pooling layer for title
    title_pooling = GlobalMaxPooling1D()(title_bilstm)
    # Global Max Pooling layer for text
    text_pooling = GlobalMaxPooling1D()(text_bilstm)

    # Concatenate title and text pooling layers
    concatenated_pooling = concatenate([title_pooling, text_pooling])

    # Dense layer for final prediction
    output_layer = Dense(5, activation='softmax')(concatenated_pooling)

    # Create model
    model = Model(inputs=[title_input, text_input], outputs=output_layer)

    return model

# Build the BiLSTM model
model_BiLSTM = build_bilstm_model()
model_BiLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_BiLSTM.summary()

model_BiLSTM.save_weights('BiLSTM_text_classification.h5')

history = model_BiLSTM.fit(
    [np.array(train_title_ids), np.array(train_text_ids)],
    y_train,
    epochs=EPOCHS,
    batch_size=4,
    verbose=1,
    validation_data=([np.array(val_title_ids), np.array(val_text_ids)], y_val)
)
# Dự đoán trên tập validation
val_pred = model_BiLSTM.predict([np.array(val_title_ids), np.array(val_text_ids)])

# Chuyển đổi dự đoán về dạng categorical
val_pred_categorical = np.argmax(val_pred, axis=1)

# Tính các metrics
report = classification_report(np.argmax(y_val, axis=1), val_pred_categorical)
print(report)
xây dựng hàm đánh giá
def test_BiLSTM(X_test, y_test):
    y_pred = model_BiLSTM.predict(X_test)
    pred = np.argmax(y_pred,axis=1)

    print(classification_report(y_test, pred))

# đánh giá trên tập test
test_BiLSTM([np.array(test_title_ids), np.array(test_text_ids)], np.array(test_labels))


# Ensemble of LSTM+CNN
def build_ensemble_model(model_LSTM, model_CNN):
    # Define input layers for the title and text inputs
    title_input = Input(shape=(train_title_ids.shape[1],))
    text_input = Input(shape=(train_text_ids.shape[1],))

    # Get the LSTM and CNN predictions
    lstm_predictions = model_LSTM([title_input, text_input])
    cnn_predictions = model_CNN([title_input, text_input])

    # Select the predictions for labels 1 to 5
    lstm_selected = lstm_predictions[:, :5]
    cnn_selected = cnn_predictions[:, :5]

    # Concatenate the selected predictions
    concatenated_predictions = Concatenate()([lstm_selected, cnn_selected])

    # Calculate the average of concatenated predictions
    ensemble_predictions = Dense(5, activation='softmax')(concatenated_predictions)

    ensemble_model = Model(inputs=[title_input, text_input], outputs=ensemble_predictions)

    return ensemble_model

# Build the ensemble model
model_ensemble_cnn_lstm = build_ensemble_model(model_LSTM, model_CNN)
model_ensemble_cnn_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_ensemble_cnn_lstm.summary()
model_ensemble_cnn_lstm.save_weights('LSTM_CNN_text_classification.h5')
history = model_ensemble_cnn_lstm.fit(
    [np.array(train_title_ids), np.array(train_text_ids)],
    y_train,
    epochs=EPOCHS,
    batch_size=4,
    verbose=1,
    validation_data=([np.array(val_title_ids), np.array(val_text_ids)], y_val)
)

# Dự đoán trên tập validation
val_pred = model_ensemble_cnn_lstm.predict([np.array(val_title_ids), np.array(val_text_ids)])

# Chuyển đổi dự đoán về dạng categorical
val_pred_categorical = np.argmax(val_pred, axis=1)

# Tính các metrics
report = classification_report(np.argmax(y_val, axis=1), val_pred_categorical)
print(report)
# xây dựng hàm đánh giá
def test_LSTM_CNN(X_test, y_test):
    y_pred = model_ensemble_cnn_lstm.predict(X_test)
    pred = np.argmax(y_pred,axis=1)

    print(classification_report(y_test, pred))

# đánh giá trên tập test
test_LSTM_CNN([np.array(test_title_ids), np.array(test_text_ids)], np.array(test_labels))


# Ensemble of BiLSTM+CNN
def build_ensemble_model(model_BiLSTM, model_CNN):
    # Define input layers for the title and text inputs
    title_input = Input(shape=(train_title_ids.shape[1],))
    text_input = Input(shape=(train_text_ids.shape[1],))

    # Get the predictions from the BiLSTM model
    lstm_predictions = model_BiLSTM([title_input, text_input])

    # Get the predictions from the CNN model
    cnn_predictions = model_CNN([title_input, text_input])

    # Concatenate the predictions
    concatenated_predictions = Concatenate()([lstm_predictions, cnn_predictions])

    # Add a dense layer
    dense_layer = Dense(64, activation='relu')(concatenated_predictions)

    # Add another dense layer for the final output
    output_layer = Dense(5, activation='softmax')(dense_layer)

    ensemble_model = Model(inputs=[title_input, text_input], outputs=output_layer)

    return ensemble_model

# Build the ensemble model
model_ensemble_bilstm_cnn = build_ensemble_model(model_BiLSTM, model_CNN)
model_ensemble_bilstm_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ensemble_bilstm_cnn.summary()

model_ensemble_bilstm_cnn.save_weights('BiLSTM_CNN_text_classification.h5')
history = model_ensemble_bilstm_cnn.fit(
    [np.array(train_title_ids), np.array(train_text_ids)],
    y_train,
    epochs=EPOCHS,
    batch_size=4,
    verbose=1,
    validation_data=([np.array(val_title_ids), np.array(val_text_ids)], y_val)
)
# Dự đoán trên tập validation
val_pred = model_ensemble_bilstm_cnn.predict([np.array(val_title_ids), np.array(val_text_ids)])

# Chuyển đổi dự đoán về dạng categorical
val_pred_categorical = np.argmax(val_pred, axis=1)

# Tính các metrics
report = classification_report(np.argmax(y_val, axis=1), val_pred_categorical)
print(report)

# xây dựng hàm đánh giá
def test_BiLSTM_CNN(X_test, y_test):
    y_pred = model_ensemble_bilstm_cnn.predict(X_test)
    pred = np.argmax(y_pred,axis=1)

    print(classification_report(y_test, pred))

# đánh giá trên tập test
test_BiLSTM_CNN([np.array(test_title_ids), np.array(test_text_ids)], np.array(test_labels))
