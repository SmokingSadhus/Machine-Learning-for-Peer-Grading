import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from cnn import TextCNN
from tensorflow.contrib import learn


x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
