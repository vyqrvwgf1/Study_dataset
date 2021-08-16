import tensorflow as tf
import numpy as np

class TextConfig():

    embed_size=100     #dimension of word embedding
    
    pre_trianing = None   #use vector_char trained by word2vec

    num_filters=128        #number of convolution kernel
    filter_sizes=[2,3,4]   #size of convolution kernel


    keep_prob=0.5          #droppout
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
    out_dir= file_dir+'/textcnn/'
    train_out_dir=file_dir+'/textcnn/'+num.strip()+'/prediction.csv' # path of evaluaion results 
    vocab_filename=file_dir+'/vocab/vocab'        #vocabulary
    vector_word_filename=file_dir+'/vocab/vocabvector_word.txt'  #vector_word trained by word2vec
    vector_word_npz=file_dir+'/vocab/word2vec.npz'   # save vector_word to numpy file

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, config,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.get_variable("W", shape=[config.vocab_size, config.embed_size],
                                             initializer=tf.constant_initializer(config.pre_trianing))
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(config.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):

                    filter_shape = [filter_size, config.embed_size, 1, config.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[config.num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

        # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.final_output = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            fc_w = tf.get_variable(
                "fc_w",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            fc_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="fc_b")
            l2_loss += tf.nn.l2_loss(fc_w)
            l2_loss += tf.nn.l2_loss(fc_b)
            self.scores = tf.nn.xw_plus_b(self.final_output, fc_w, fc_b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(config.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.Variable(0, trainable=False, name='global_step'))

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")