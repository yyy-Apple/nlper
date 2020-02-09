import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import os
from itertools import islice

# Global Argument:
load_model = False


# change the original file into dataframe
def preprocess(datapath):
    data_dict = {'words': [], 'tags': []}
    with open(datapath, 'r') as f:
        for line in f:
            tag, word = line.lower().strip().split(" ||| ")
            data_dict['words'].append(word)
            data_dict['tags'].append(tag)
    return pd.DataFrame(data_dict)


train_df = preprocess("./topicclass/topicclass_train.txt")
val_df = preprocess("./topicclass/topicclass_valid.txt")
test_df = preprocess("./topicclass/topicclass_test.txt")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
gpus = [0]

for index, row in val_df.iterrows():
    # print(row)
    if row.tags == "media and darama":
        val_df.at[index, 'tags'] = "media and drama"


class Vocabulary(object):
    '''
    A super class for word vocabulary and be directly used for tag vocabulary
    '''

    def __init__(self, word2index=None):
        self.word2index = word2index
        if self.word2index is None:
            self.word2index = {}
            self.index2word = {}
        else:
            self.index2word = {v: k for k, v in self.word2index}

    def add_word(self, word):
        if word in self.word2index.keys():
            index = self.word2index[word]
        else:
            index = len(self.word2index)
            self.word2index[word] = index
            self.index2word[index] = word
        return index

    def lookup_word(self, word):
        return self.word2index.get(word, -1)

    def __len__(self):
        return len(self.word2index)

    def __str__(self):
        return "This is a vocabulary, it's size is %d" % len(self)


class WordVocabulary(Vocabulary):
    '''
    This is a word vocabulary including <UNK> and <PAD>
    <UNK> -- used for rare words
    <PAD> -- used for padding the sentences in order to batch them together
    '''

    def __init__(self, word2index=None):
        super().__init__()
        self.word2index = word2index
        if self.word2index is None:
            self.word2index = {}
        self.unk_index = self.add_word('<UNK>')
        self.pad_index = self.add_word('<PAD>')

    def lookup_word(self, word):
        return self.word2index.get(word, self.unk_index)


class Vectorizer(object):
    '''
    It has word vocabulary and tag vocabulary
    '''

    def __init__(self, wordVocabulary, tagVocabulary):
        self.wordVocabulary = wordVocabulary
        self.tagVocabulary = tagVocabulary

    def vectorize(self, sentence, max_seq_length):
        # because this is used for index, it should be int
        result = np.zeros(max_seq_length, dtype=np.int64)
        sentence = sentence.split(" ")
        word_indices = [self.wordVocabulary.lookup_word(word) for word in sentence]
        result[: len(word_indices)] = word_indices
        result[len(word_indices):] = self.wordVocabulary.pad_index
        return result

    @classmethod
    def create_Vectorizer(cls, train_df, cutoff):
        '''
        cutoff -- when creating the vectorizer,  rare words will not be add to the word vocabulary
        '''
        wordVocabulary = WordVocabulary()
        tagVocabulary = Vocabulary()

        for tag in sorted(set(train_df.tags)):
            tagVocabulary.add_word(tag)

        wordCounter = {}
        for sentence in train_df.words:
            for word in sentence.split(" "):
                if word not in wordCounter.keys():
                    wordCounter[word] = 1
                else:
                    wordCounter[word] += 1

        # only keep those words that appear more than cutoff value
        for word, count in wordCounter.items():
            if count >= cutoff:
                wordVocabulary.add_word(word)

        return cls(wordVocabulary, tagVocabulary)


class MyDataset(Dataset):
    '''
    pack all the train_df, val_df and test_df together
    '''

    def __init__(self, train_df, val_df, test_df, vectorizer):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.all_df = pd.concat([self.train_df, self.val_df, self.test_df])

        # which data are we working on
        self.target_df = self.train_df

        self.vectorizer = vectorizer

        get_length = lambda context: len(context.split(" "))
        self.max_seq_length = max(map(get_length, self.all_df.words))

        # also, there exists imbalance problem.. This is used in the CrossEntropy loss
        # only count train and val tags because the tags in the test dataset is all unk
        class_counts = pd.concat([self.train_df, self.val_df]).tags.value_counts().to_dict()

        def sort_key(item):
            return self.vectorizer.tagVocabulary.lookup_word(item[0])

        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for item, count in sorted_counts]

        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def create_Dataset(cls, train_df, val_df, test_df, cutoff):
        return cls(train_df, val_df, test_df, Vectorizer.create_Vectorizer(train_df, cutoff))

    def set_target_df(self, target):
        if target == "train":
            self.target_df = self.train_df
        elif target == "val":
            self.target_df = self.val_df
        elif target == "test":
            self.target_df = self.test_df
        else:
            raise KeyError("There is no such target: %s" % target)

    def __len__(self):
        return len(self.target_df)

    def __getitem__(self, index):
        item = self.target_df.iloc[index]

        words_vector = self.vectorizer.vectorize(item.words, self.max_seq_length)
        tag_vector = self.vectorizer.tagVocabulary.lookup_word(item.tags)

        return {'x': words_vector, 'y': tag_vector}


myDataset = MyDataset.create_Dataset(train_df, val_df, test_df, 2)

if not os.path.exists("train_embeddings.npy"):
    # load word vectors and make embedding matrix
    # first convert it to a matrix
    word2index = {}
    embeddings = []
    with open("./topicclass/GoogleNews-vectors-negative300.txt", 'r') as f:
        for lineno, line in enumerate(islice(f, 1, None)):
            if (lineno % 10000 == 0):
                print("on %d word" % lineno)
            line = line.split(" ")
            word2index[line[0]] = lineno
            embeddings.append([float(number) for number in line[1:]])

    embeddings = np.array(embeddings)

    # Then use all the words in the training set to construct the corresponding word embeddings
    train_embeddings = []
    for word in myDataset.vectorizer.wordVocabulary.word2index.keys():
        if word in word2index.keys():
            index = word2index[word]  # index in the embeddings
            train_embeddings.append(embeddings[index])
        else:
            embedding_i = torch.ones(1, len(embeddings[0]))
            torch.nn.init.xavier_uniform_(embedding_i)
            embedding_i = embedding_i.view(-1)
            train_embeddings.append(embedding_i)

    train_embeddings = np.stack(train_embeddings)
    np.save("train_embeddings.npy", train_embeddings)
else:
    train_embeddings = np.load("train_embeddings.npy")


class CNNClassifier(nn.Module):
    '''
    1 convnet layers followed by 1 fully connected layers
    '''

    def __init__(self, pretrained_embeddings, num_embeddings, embedding_dim, channels, hidden_units, num_classes,
                 dropout_prob):
        super().__init__()
        weight = torch.from_numpy(pretrained_embeddings).float()
        # self.embedding = nn.Embedding(num_embeddings = num_embeddings, embedding_dim=embedding_dim, _weight = weight)
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)  #
        # convnet --  1 layer, window size of 3, 4, 5
        for i in range(3):
            convnet = nn.Conv1d(in_channels=embedding_dim, out_channels=channels, kernel_size=i + 3)
            nn.init.kaiming_normal_(convnet.weight)  # xavier_uniform_, gain = 1
            setattr(self, f'convnet{i}', convnet)

        self.dropout_prob = dropout_prob

        # mlp
        self.fc1 = nn.Linear(channels * 3, num_classes)
        nn.init.kaiming_normal_(self.fc1.weight)  # xavier_uniform_, gain = 1

    def forward(self, x_in, apply_softmax=False):
        # go through the embedding lookup
        output = self.embedding(x_in).permute(0, 2, 1)
        # add a dropout
        # output = F.dropout(output, p = 0.2)

        # go through the convnet and maxpooling

        convnet_output = []
        for i in range(3):
            intermediate_output = F.relu(getattr(self, f'convnet{i}')(output))
            convnet_output.append(F.max_pool1d(intermediate_output, intermediate_output.size(dim=2)).squeeze(dim=2))
        output = torch.cat(convnet_output, dim=1)
        output = F.dropout(output, p=self.dropout_prob)

        # then go through the MLP
        output = self.fc1(output)
        # output = F.dropout(output, p = self.dropout_prob)
        # output = F.relu(output)

        if apply_softmax == True:
            output = F.softmax(output, dim=1)
        # output = F.softmax(output, dim=1)
        return output


def compute_acc(y_hat, y):
    y_values, y_indices = y_hat.max(dim=1)
    n_correct = torch.eq(y_indices, y).sum().item()
    return n_correct / len(y_values) * 100


cnnClassifier = CNNClassifier(train_embeddings, train_embeddings.shape[0], train_embeddings.shape[1],
                              300, 100, len(myDataset.vectorizer.tagVocabulary), 0.5)
loss_func = nn.CrossEntropyLoss(myDataset.class_weights)

# unfreeze the embedding and fine-tine the model
if (load_model):
    cnnClassifier = torch.nn.DataParallel(cnnClassifier)
    cnnClassifier.load_state_dict(torch.load('cnnClassifier.pth'))
    emb_weight = torch.from_numpy(train_embeddings).float()
    cnnClassifier.embedding = nn.Embedding.from_pretrained(emb_weight, freeze=False)

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Use GPU')
    cnnClassifier = torch.nn.DataParallel(cnnClassifier, device_ids=gpus).cuda()
    loss_func = loss_func.cuda()

optimizer = optim.Adam(cnnClassifier.parameters(), lr=0.00003)

###################### for training ##########################

def train():
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_epoch = 0
    best_acc = 0.0
    early_stop_step = 0  # if it is greater than 5, then stop
    early_stop = False

    start = time.time()

    for epoch in range(100):
        if early_stop:
            break

        # train on training data
        myDataset.set_target_df("train")
        dataloader = DataLoader(dataset=myDataset, batch_size=50, shuffle=True)

        train_avg_loss = 0.0
        train_avg_acc = 0.0
        val_avg_loss = 0.0
        val_avg_acc = 0.0

        cnnClassifier.train()

        for batch_index, batch_dict in enumerate(dataloader):
            # the training routine is in these 5 steps:
            batch_dict_x = batch_dict['x'].cuda() if use_cuda else batch_dict['x']
            batch_dict_y = batch_dict['y'].cuda() if use_cuda else batch_dict['y']
            # --> step 1: zero the gradients
            optimizer.zero_grad()

            # --> step 2: compute the output
            y_hat = cnnClassifier(batch_dict_x)

            # --> step 3: compute the loss
            loss = loss_func(y_hat, batch_dict_y)
            loss_t = loss.item()
            train_avg_loss = (train_avg_loss * batch_index + loss_t) / (batch_index + 1)
            train_avg_acc = (train_avg_acc * batch_index + compute_acc(y_hat, batch_dict_y)) / (batch_index + 1)

            # --> step 4: use loss to produce gradients
            loss.backward()

            # --> step 5: use optimizer to take gradient step
            optimizer.step()
        train_loss.append(train_avg_loss)
        train_acc.append(train_avg_acc)
        print("on epoch: %d [train], avg_loss is %f, avg_acc is %f" % (epoch, train_avg_loss, train_avg_acc))

        # evaluate on val data
        myDataset.set_target_df("val")
        dataloader = DataLoader(dataset=myDataset, batch_size=2000)
        cnnClassifier.eval()

        for batch_index, batch_dict in enumerate(dataloader):
            batch_dict_x = batch_dict['x'].cuda() if use_cuda else batch_dict['x']
            batch_dict_y = batch_dict['y'].cuda() if use_cuda else batch_dict['y']
            y_hat = cnnClassifier(batch_dict_x)
            loss = loss_func(y_hat, batch_dict_y)
            loss_t = loss.item()
            val_avg_loss = (val_avg_loss * batch_index + loss_t) / (batch_index + 1)
            val_avg_acc = (val_avg_acc * batch_index + compute_acc(y_hat, batch_dict_y)) / (batch_index + 1)

        val_loss.append(val_avg_loss)
        val_acc.append(val_avg_acc)
        print("on epoch: %d [val], avg_loss is %f, avg_acc is %f" % (epoch, val_avg_loss, val_avg_acc))

        # save the best model and apply early stop if the accuracy on validation set worsens for a while
        if (epoch >= 1):
            second_last, last = val_acc[-2:]
            if (last < second_last):
                early_stop_step += 1
            else:
                if (last > best_acc):  # if this is the lowest loss ever
                    best_acc = last
                    best_epoch = epoch
                    torch.save(cnnClassifier.state_dict(), 'cnnClassifier.pth')
                early_stop_step = 0

        if (early_stop_step >= 4):
            early_stop = True

        stop = time.time()
        spend_time = stop - start

        print("total time spent: %d" % spend_time)

    print("best acc is on val:", best_acc)
    print("best epoch is:", best_epoch)


################################################
#     Load the best model to do prediction     #
################################################


