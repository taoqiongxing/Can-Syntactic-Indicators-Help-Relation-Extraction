import numpy as np
import pandas as pd
import nltk
import re

import utils
from configure import FLAGS
import keras
from nltk.stem import SnowballStemmer
Stemmer = SnowballStemmer('english')

nltk.download('averaged_perceptron_tagger')

def relation_shorten(short_text):
    tokens = nltk.word_tokenize(short_text)
    pos_tagged_tokens = nltk.pos_tag(tokens)

    # 1.简化并列词
    j = len(pos_tagged_tokens) - 2
    while j > 0:
        if pos_tagged_tokens[j][1] == 'CC':
            del pos_tagged_tokens[j]
            del tokens[j]
            if j < len(pos_tagged_tokens) - 1:
                del pos_tagged_tokens[j]
                del tokens[j]
        j -= 1

    # 2.保留动词，名词，IN类型，to类型，人称代词和占有代词
    rvbin = []
    rpos = []
    rvbin.append(tokens[0])
    rpos.append(pos_tagged_tokens[0])
    for k in range(1, len(tokens) - 1):
        if pos_tagged_tokens[k][1] == 'TO' or 'VB' in pos_tagged_tokens[k][1] or pos_tagged_tokens[k][
            1] == 'IN' or 'NN' in pos_tagged_tokens[k][1] or 'PRP' in pos_tagged_tokens[k][1]:
            rvbin.append(tokens[k])
            rpos.append(pos_tagged_tokens[k])
    rvbin.append(tokens[-1])
    rpos.append(pos_tagged_tokens[-1])

    # 3.简化名词
    j = len(rvbin) - 2
    while j > 0:
        if 'NN' in rpos[j][1]:
            if rpos[j + 1][1] == 'NN' or rpos[j + 1][1] == 'NNS':
                del rpos[j]
                del rvbin[j]
        j -= 1

    if len(rvbin) > 2 and 'NN' in rpos[1][1]:
        del rpos[1]
        del rvbin[1]

    # 4.简化两个实体的描述
    j = len(rvbin) - 3
    while j > 1:
        if rpos[j][1] == 'NN' or rpos[j][1] == 'NNS':
            if rpos[j + 1][0] == 'of' or rpos[j + 1][0] == 'for':
                d = j
                while d < len(rpos) - 1:
                    del rpos[d]
                    del rvbin[d]
            elif rpos[j - 1][0] == 'of' or rpos[j - 1][0] == 'for':
                d = j
                while d > 0 and d < len(rpos) - 1:
                    del rpos[d]
                    del rvbin[d]
                    d -= 1
                    j -= 1
        j -= 1

    #5.保留靠近e2的动作
    j = len(rvbin) - 2
    while j > 0:
        if rpos[j][1] == 'NN' or rpos[j][1] == 'NNS':
            d = j
            while d > 0:
                del rpos[d]
                del rvbin[d]
                d -= 1
                j -= 1
        j -= 1

    p = []
    for w in range(len(rpos)):
        if w == 0 or w == len(rpos) - 1:
            if 'NN' not in rpos[w][1]:
                p.append('NN')
            else:
                p.append(rpos[w][1])
        else:
            p.append(rpos[w][1])
    p = " ".join(p)

    return r, p, add_rw_cate

def relation_words(between_e):
    words=[]
    for i in between_e[0]:
        i=i.split()
        for j in i:
            w=Stemmer.stem(j)
            words.append(w)
    for i in between_e[1]:
        i=i.split()
        for j in i:
            w=Stemmer.stem(j)
            words.append(w)
    words=pd.value_counts(words)
    return words

def relation_words_between_entity(between_e,words):
    rwbe=[]
    for i in between_e:
        wordslist=i.split()
        rwbe_i=[]
        for j in wordslist:
            num_the=0
            w=Stemmer.stem(j)
            if words[w]>30 and j!='-' and j!='a' and j!='an' and j!='an':
                if j!='the':
                    rwbe_i.append(j)
                else:
                    num_the+=1
                if num_the>1:
                    rwbe_i=[]
        if len(rwbe_i)==0:
            rwbe_i.append('NANA')
        rwbe_i=rwbe_i[-4:]
        rwbe_i=" ".join(rwbe_i)
        rwbe.append(rwbe_i)
    return rwbe

def clean_str(text):
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def load_data_and_labels(path):
    data = []
    lines = [line.strip() for line in open(path)]
    max_sentence_length = 0
    for idx in range(0, len(lines), 4):
        id = lines[idx].split("\t")[0]
        relation = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][1:-1]
        sentence = sentence.replace('<e1>', ' _e11_ ')
        sentence = sentence.replace('</e1>', ' _e12_ ')
        sentence = sentence.replace('<e2>', ' _e21_ ')
        sentence = sentence.replace('</e2>', ' _e22_ ')

        sentence = clean_str(sentence)
        tokens = nltk.word_tokenize(sentence)
        if max_sentence_length < len(tokens):
            max_sentence_length = len(tokens)
        e1 = tokens.index("e12") - 1
        e2 = tokens.index("e22") - 1
        sentence = " ".join(tokens)

        # 两个实体之间的词
        between_e = tokens[tokens.index("e11"):tokens.index("e22")+1]
        between_e = " ".join(between_e)

        between_e = between_e.replace('e11 ', '')
        between_e = between_e.replace(' e12', '')
        between_e = between_e.replace('e21 ', '')
        between_e = between_e.replace(' e22', '')

        relationword, relationwordpos, add_rw_cate = relation_shorten(between_e)

        data.append([id, sentence, e1, e2, relation, relationword, relationwordpos, add_rw_cate])

    print(path)
    print("max sentence length = {}\n".format(max_sentence_length))

    df = pd.DataFrame(data=data, columns=["id", "sentence", "e1", "e2", "relation", "relationword", "relationwordpos", "rw_cate"])

    pos1, pos2 = get_relative_position(df, FLAGS.max_sentence_length)

    df['label'] = [utils.class2label[r] for r in df['relation']]

    # Text Data
    x_text = df['sentence'].tolist()

    relationword = df['relationword'].tolist()
    relationwordpos = df['relationwordpos'].tolist()
    rw_cate = df['rw_cate'].tolist()


    e1 = df['e1'].tolist()
    e2 = df['e2'].tolist()

    # Label Data
    y = df['label']
    labels_flat = y.values.ravel()
    labels_count = np.unique(labels_flat).shape[0]

    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0 0 ... 0 0 0 0 0]
    # 1  => [0 1 0 0 0 ... 0 0 0 0 0]
    # ...
    # 18 => [0 0 0 0 0 ... 0 0 0 0 1]
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    return x_text, labels, e1, e2, pos1, pos2, relationword, relationwordpos, rw_cate


def get_relative_position(df, max_sentence_length):
    # Position data
    pos1 = []
    pos2 = []
    for df_idx in range(len(df)):
        sentence = df.iloc[df_idx]['sentence']
        tokens = nltk.word_tokenize(sentence)
        e1 = df.iloc[df_idx]['e1']
        e2 = df.iloc[df_idx]['e2']

        p1 = ""
        p2 = ""
        for word_idx in range(len(tokens)):
            p1 += str((max_sentence_length - 1) + word_idx - e1) + " "
            p2 += str((max_sentence_length - 1) + word_idx - e2) + " "
        pos1.append(p1)
        pos2.append(p2)

    return pos1, pos2


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == "__main__":
    trainFile = 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    testFile = 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'

    load_data_and_labels(testFile)
