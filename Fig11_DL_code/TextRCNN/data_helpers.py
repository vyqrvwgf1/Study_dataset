import numpy as np
import re
from collections import  Counter
import tensorflow.contrib.keras as kr
import numpy as np
import codecs
import re
import nltk
import string
from nltk.corpus import stopwords         #停用词
from nltk.tokenize import word_tokenize   #分词
from nltk.stem import PorterStemmer       #词干化
from nltk.stem import WordNetLemmatizer   #词形还原

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def read_file_into_processed(filename):
    """
    Args:
        filename:trian_filename,test_filename,val_filename 
    Returns:
        two list where the first is lables and the second is contents cut by jieba
        
    """


    content = list(open(filename, "r", encoding='utf-8').readlines())
    content = [s.strip() for s in content]

    stop_words = set(stopwords.words('english'))  #英文停用分词集合

    sen_list=[]

    for sents in content:
        remove= string.punctuation
        table=str.maketrans('','',remove)
        newsent=sents.translate(table)
        fenci=word_tokenize(newsent)
        filtered_sentence = [w for w in fenci if w not in stop_words]
        sen_list.append(filtered_sentence)
    stem_list=[]  
    ps =PorterStemmer()
    for w in sen_list: 
        stem_words = []
        for k in w:
            rootWord=ps.stem(k) 
            stem_words.append(rootWord)
        stem_list.append(stem_words)


    lemma_list = []
    wordnet_lemmatizer = WordNetLemmatizer()

    for w in stem_list: 
        lemma_word = []
        for k in w:
            word1 = wordnet_lemmatizer.lemmatize(k, pos = "n") 
            word2 = wordnet_lemmatizer.lemmatize(word1, pos = "v") 
            word3 = wordnet_lemmatizer.lemmatize(word2, pos = ("a")) 
            lemma_word.append(word3)
        lemma_list.append(lemma_word)
    return lemma_list

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
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
def read_vocab(vocab_dir):
    """
    Args:
        filename:path of vocab_filename
    Returns:
        words: a list of vocab
        word_to_id: a dict of word to id
        
    """
    words=codecs.open(vocab_dir,'r',encoding='utf-8').read().strip().split('\n')
    word_to_id=dict(zip(words,range(len(words))))
    return words,word_to_id

def read_category():
    """
    Args:
        None
    Returns:
        categories: a list of label
        cat_to_id: a dict of label to id

    """
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id=dict(zip(categories,range(len(categories))))
    return categories,cat_to_id

def process_file(filename,word_to_id,cat_to_id,max_length=600):
    """
    Args:
        filename:train_filename or test_filename or val_filename
        word_to_id:get from def read_vocab()
        cat_to_id:get from def read_category()
        max_length:allow max length of sentence 
    Returns:
        x_pad: sequence data from  preprocessing sentence 
        y_pad: sequence data from preprocessing label

    """
    labels,contents=read_file(filename)
    data_id,label_id=[],[]
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    x_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length,padding='post', truncating='post')
    y_pad=kr.utils.to_categorical(label_id)
    return x_pad,y_pad


def export_word2vec_vectors(vocab, word2vec_dir,trimmed_filename):
    """
    Args:
        vocab: word_to_id 
        word2vec_dir:file path of have trained word vector by word2vec
        trimmed_filename:file path of changing word_vector to numpy file
    Returns:
        save vocab_vector to numpy file
        
    """
    file_r = codecs.open(word2vec_dir, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()
    np.savez_compressed(trimmed_filename, embeddings=embeddings)

def get_training_word2vec_vectors(filename):
    """
    Args:
        filename:numpy file
    Returns:
        data["embeddings"]: a matrix of vocab vector
    """
    with np.load(filename) as data:
        return data["embeddings"]

def build_vocab(data_1,data_2,vocab_dir,vocab_size=18000):
    """
    Args:
        filename:trian_filename,test_filename,val_filename
        vocab_dir:path of vocab_filename
        vocab_size:number of vocabulary
    Returns:
        writting vocab to vocab_filename

    """
    all_data = []
    for content in data_1:
        all_data.extend(content)
    for content in data_2:
        all_data.extend(content)
    counter=Counter(all_data)
    count_pairs=counter.most_common(vocab_size-1)
    words,_=list(zip(*count_pairs))
    words=['<PAD>']+list(words)

    with codecs.open(vocab_dir,'w',encoding='utf-8') as f:
        f.write('\n'.join(words)+'\n')