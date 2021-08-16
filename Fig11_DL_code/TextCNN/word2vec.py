# 引入 word2vec
from gensim.models import word2vec
import nltk
import gensim
from nltk.corpus import stopwords         #停用词
from nltk.tokenize import word_tokenize   #分词
from nltk.stem import PorterStemmer       #词干化
from nltk.stem import WordNetLemmatizer   #词形还原
import numpy as np
import tensorflow as tf
import data_helpers as dh
# 引入日志配置
import logging
from tensorflow.contrib import learn

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 引入数据集
raw_sentences = ["the quick brown fox jumps over the lazy dogs","yoyoyo you go home now to sleep"]

# 英文分词
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#positive_examples = list(open('F:/pdf/cnn-text-classification-tf-master/data/rt-polaritydata/rt-polarity.pos', "r", encoding='utf-8').readlines())
#positive_examples = [s.strip() for s in positive_examples]
#negative_examples = list(open('F:/pdf/cnn-text-classification-tf-master/data/rt-polaritydata/rt-polarity.neg', "r", encoding='utf-8').readlines())
#negative_examples = [s.strip() for s in negative_examples]
    # Split by words
#x_text = positive_examples + negative_examples

#分词
x_text = dh.read_file_into_processed('F:/pdf/cnn-text-classification-tf-master/data/word2vec_train.txt')
#x_text_2 = dh.read_file_into_processed('F:/pdf/cnn-text-classification-tf-master/data/rt-polaritydata/rt-polarity.neg')
#x_text=x_text_1+x_text_2
#构建词典
#vocab_dir = 'F:/pdf/cnn-text-classification-tf-master/data/vocab'
#dh.build_vocab(x_text,x_text,vocab_dir,20000)
#vocab_processor = learn.preprocessing.VocabularyProcessor(30)
#words,word_to_id = dh.read_vocab(vocab_dir)
#onehot = np.array(list(vocab_processor.fit_transform(x_text)))
ci=0
line=0
for x in x_text:
	ci+=len(x)
	line+=1
	pass
print(ci/line)



# 构建模型
#word2vec_dir='F:/pdf/cnn-text-classification-tf-master/data/word2vec.txt'
#vector_word_npz='F:/pdf/cnn-text-classification-tf-master/data/word2vec.npz'

#model = word2vec.Word2Vec(x_text, min_count=1)
#model.wv.save_word2vec_format(word2vec_dir, binary=False)
#dh.export_word2vec_vectors(word_to_id, word2vec_dir, vector_word_npz)

#model=gensim.models.KeyedVectors.load_word2vec_format('F:/pdf/cnn-text-classification-tf-master/data/word2vec.txt',binary=False)

#model.save('F:/pdf/cnn-text-classification-tf-master/data/word2vec')
#export_word2vec_vectors(word_to_id, config.vector_word_filename, config.vector_word_npz)


# 进行相关性比较sen
