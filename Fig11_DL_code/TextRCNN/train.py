import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers as dh
import data_process
import tensorflow.contrib.keras as kr
from rcnn import *
from rcnn import RCNN
from rcnn import TextConfig
from tensorflow.contrib import learn

# define parameters


#configuration
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_integer("num_epochs", 100, "embedding size")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.") #batch_ size 32-->128

tf.flags.DEFINE_integer("decay_steps", 12000, "how many steps before decay learning rate.")
tf.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.")  

tf.flags.DEFINE_string("ckpt_dir", "text_rcnn_checkpoint/", "checkpoint location for the model")
tf.flags.DEFINE_integer('num_checkpoints', 10, 'save checkpoints count')

tf.flags.DEFINE_integer("sequence_length", 300, "max sentence length")
tf.flags.DEFINE_integer("embed_size", 128, "embedding size")
tf.flags.DEFINE_integer('context_size',128,'word contenxt size')
tf.flags.DEFINE_integer('hidden_size', 128, 'cell output size')

tf.flags.DEFINE_boolean("is_training", True, "is traning.true:tranining,false:testing/inference")

tf.flags.DEFINE_integer("validate_every", 100, "Validate every validate_every epochs.") 
tf.flags.DEFINE_float('validation_percentage',0.1,'validat data percentage in train data')
tf.flags.DEFINE_integer('dev_sample_max_cnt', 1000, 'max cnt of validation samples, dev samples cnt too large will case high loader')

tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularization lambda (default: 0.0)")

tf.flags.DEFINE_float('grad_clip', 5.0, 'grad_clip')

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def prepocess():
    """
    For load and process data
    :return:
    """
   # Load data
    config = TextConfig()
    print("Loading data...")
    x_text_1 = dh.read_file_into_processed(config.train_pos_filename)
    x_text_2 = dh.read_file_into_processed(config.train_neg_filename)
    x_text = x_text_1+x_text_2

    positive_labels = [[0, 1] for _ in x_text_1]
    negative_labels = [[1, 0] for _ in x_text_2]
    y = np.concatenate([positive_labels, negative_labels], 0)

    # Build vocabulary
    max_document_length = max([len(x) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    
    words,word_to_id = dh.read_vocab(config.vocab_filename)
    config.vocab_size = len(words)
    config.pre_trianing = dh.get_training_word2vec_vectors(config.vector_word_npz)

    # shuffle
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))

    data_id=[]
    for i in range(len(x_text)):
        data_id.append([word_to_id[x] for x in x_text[i] if x in word_to_id])

    x=kr.preprocessing.sequence.pad_sequences(data_id,30,padding='post', truncating='post')

    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # split train/test dataset
    x_train, x_dev = x_shuffled, x_shuffled
    y_train, y_dev = y_shuffled, y_shuffled

    del x, y, x_shuffled, y_shuffled

    print('Vocabulary Size: {:d}'.format(len(word_to_id)))
    print('Train/Dev split: {:d}/{:d}'.format(len(y_train), len(y_dev)))
    return x_train, y_train, config, x_dev, y_dev

def train(x_train, y_train, config1, x_dev, y_dev):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            # allows TensorFlow to fall back on a device with a certain operation implemented
            allow_soft_placement= FLAGS.allow_soft_placement,
            # allows TensorFlow log on which devices (CPU or GPU) it places operations
            log_device_placement=FLAGS.log_device_placement
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # initialize rcnn
            rcnn = RCNN(sequence_length=x_train.shape[1],
                        num_classes=y_train.shape[1],
                        vocab_size=1,
                        embed_size=FLAGS.embed_size,
                        l2_lambda=FLAGS.l2_reg_lambda,
                        grad_clip=FLAGS.grad_clip,
                        learning_rate=FLAGS.learning_rate,
                        decay_steps=FLAGS.decay_steps,
                        decay_rate=FLAGS.decay_rate,
                        hidden_size=FLAGS.hidden_size,
                        context_size=FLAGS.context_size,
                        activation_func=tf.tanh,
                        config=config1
                        )


            # output dir for models and summaries
            timestamp = str(time.time())
            out_dir = os.path.join(config1.out_dir, config1.num.strip())
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print('Writing to {} \n'.format(out_dir))

            # checkpoint dir. checkpointing – saving the parameters of your model to restore them later on.
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            #vocab_processor.save(os.path.join(out_dir, 'vocab'))

            # Initialize all
            sess.run(tf.global_variables_initializer())


            def train_step(x_batch, y_batch):
                """
                A single training step
                :param x_batch:
                :param y_batch:
                :return:
                """
                feed_dict = {
                    rcnn.input_x: x_batch,
                    rcnn.input_y: y_batch,
                    rcnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss, accuracy = sess.run(
                    [rcnn.train_op, rcnn.global_step, rcnn.loss_val, rcnn.accuracy],
                    feed_dict=feed_dict
                )
                time_str = datetime.datetime.now().isoformat()


            def dev_step(x_batch, y_batch):
                """
                Evaluate model on a dev set
                Disable dropout
                :param x_batch:
                :param y_batch:
                :param writer:
                :return:
                """
                feed_dict = {
                    rcnn.input_x: x_batch,
                    rcnn.input_y: y_batch,
                    rcnn.dropout_keep_prob: 1.0
                }
                step, loss, accuracy = sess.run(
                    [rcnn.global_step, rcnn.loss_val, rcnn.accuracy],
                    feed_dict=feed_dict
                )
                time_str = datetime.datetime.now().isoformat()
                print("dev results:{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            # generate batches
            batches = data_process.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # training loop
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, rcnn.global_step)
                if current_step % FLAGS.validate_every == 0:
                    print('\n Evaluation:')
                    dev_step(x_dev, y_dev)
                    print('')

            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print('Save model checkpoint to {} \n'.format(path))

def main(argv=None):
    x_train, y_train, config, x_dev, y_dev = prepocess()
    train(x_train, y_train, config, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()