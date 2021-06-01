# 引入 word2vec
from gensim.models import word2vec
import numpy as np
import data_helpers as dh
# 引入日志配置
import logging
from tensorflow.contrib import learn
from text_cnn import *
config=TextConfig()