#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers as dh
from fastText import *
import tensorflow.contrib.keras as kr
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================
config = TextConfig()

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# 修改
tf.flags.DEFINE_string("checkpoint_dir", os.path.join(config.out_dir, config.num.strip(),"Checkpoints"), "Checkpoint directory from training run")



tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
config = TextConfig()

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = dh.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    #x_raw = ["a masterpiece four years in the making", "everything is off."]
    #y_test = [1, 0]
    print("Loading data...")
    x_text_1 = dh.read_file_into_processed(config.test_pos_filename)
    x_text_2 = dh.read_file_into_processed(config.test_neg_filename)
    x_raw = x_text_1+x_text_2

    positive_labels = [[0, 1] for _ in x_text_1]
    negative_labels = [[1, 0] for _ in x_text_2]
    y_test = np.concatenate([positive_labels, negative_labels], 0)
    y_test = np.argmax(y_test, axis=1)


# Map data into vocabulary
words,word_to_id = dh.read_vocab(config.vocab_filename)
config.vocab_size = len(words)
config.pre_trianing = dh.get_training_word2vec_vectors(config.vector_word_npz)

data_id=[]
for i in range(len(x_raw)):
    data_id.append([word_to_id[x] for x in x_raw[i] if x in word_to_id])
x_test=kr.preprocessing.sequence.pad_sequences(data_id,30,padding='post', truncating='post')

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        #dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate

        predictions = graph.get_operation_by_name("predictions").outputs[0]
        score = graph.get_operation_by_name("score").outputs[0]

        # Generate batches for one epoch
        batches = dh.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = config.train_out_dir
print("Saving evaluation to {0}".format(out_path))
fo = open(out_path, "w")
for k in all_predictions:
    fo.write(str(k)+"\n")