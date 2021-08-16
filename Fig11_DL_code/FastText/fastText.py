from BaseModel import BaseModel
import tensorflow as tf

class TextConfig():

    embedding_size=100     #dimension of word embedding
    
    pre_trianing = None   #use vector_char trained by word2vec

    lr= 1e-3               #learning rate
    lr_decay= 0.9          #learning rate decay
    clip= 6.0              #gradient clipping threshold
    l2_reg_lambda=0.01     #l2 regularization lambda

    configfile = open("F:/ASE_code/config.txt", "r")

    num= configfile.readline()

    train_pos = configfile.readline()
    train_neg = configfile.readline()
    test_pos = configfile.readline()
    test_neg = configfile.readline()

    train_pos_filename=train_pos.strip()+num.strip() #train data
    train_neg_filename=train_neg.strip()+num.strip()
    test_pos_filename=test_pos.strip()+num.strip()   #test data
    test_neg_filename=test_neg.strip()+num.strip()
    file_dir= 'F:/ASE_code'  # modify this path according to your local path of this project
    out_dir= file_dir+'/FastText/'
    train_out_dir=file_dir+'/FastText/'+num.strip()+'/prediction.csv' # path of evaluaion results 
    vocab_filename=file_dir+'/vocab/vocab'        #vocabulary
    vector_word_filename=file_dir+'/vocab/vocabvector_word.txt'  #vector_word trained by word2vec
    vector_word_npz=file_dir+'/vocab/word2vec.npz'   # save vector_word to numpy file


class fastTextModel(BaseModel):
    """
    A simple implementation of fasttext for text classification
    """
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, learning_rate, decay_steps, decay_rate,
                 l2_reg_lambda, is_training=True,
                 initializer=tf.random_normal_initializer(stddev=0.1)):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.is_training = is_training
        self.l2_reg_lambda = l2_reg_lambda
        self.initializer = initializer

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.num_classes], name='input_y')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.instantiate_weight()
        self.logits = self.inference()
        self.loss_val = self.loss()
        self.train_op = self.train()

        self.score=tf.add(self.logits, 0, name='score')
        self.predictions = tf.argmax(self.logits, axis=1, name='predictions')
        correct_prediction = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='accuracy')

    def instantiate_weight(self):
        with tf.name_scope('weights'):
            self.Embedding = tf.get_variable('Embedding', shape=[self.vocab_size, self.embedding_size],
                                             initializer=self.initializer)
            self.W_projection = tf.get_variable('W_projection', shape=[self.embedding_size, self.num_classes],
                                                initializer=self.initializer)
            self.b_projection = tf.get_variable('b_projection', shape=[self.num_classes])


    def inference(self):
        """
        1. word embedding
        2. average embedding
        3. linear classifier
        :return:
        """
        # embedding layer
        with tf.name_scope('embedding'):
            words_embedding = tf.nn.embedding_lookup(self.Embedding, self.input_x)
            self.average_embedding = tf.reduce_mean(words_embedding, axis=1)

        logits = tf.matmul(self.average_embedding, self.W_projection) +self.b_projection

        return logits


    def loss(self):
        # loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            data_loss = tf.reduce_mean(losses)
            l2_loss = tf.add_n([tf.nn.l2_loss(cand_var) for cand_var in tf.trainable_variables()
                                if 'bias' not in cand_var.name]) * self.l2_reg_lambda
            data_loss += l2_loss * self.l2_reg_lambda
            return data_loss

    def train(self):
        with tf.name_scope('train'):
            learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                       self.decay_steps, self.decay_rate,
                                                       staircase=True)

            train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                      learning_rate=learning_rate, optimizer='Adam')

        return train_op