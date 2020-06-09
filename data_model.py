# Author: adam sigal, following Lilian Weng's blog post
import numpy as np
import os
import random
import re
import shutil
import time
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.contrib.tensorboard.plugins import projector

# self is like `model` object
class LstmRNN(object):
    def __init__(
        self,
        sess,
        stock_count,
        num_steps=30,
        input_size=1,
        embed_size=None,
        logs_dir='logs',
        plots_dir='imgs'
    ):
    """
        Construct a RNN model using LSTM cell.
        Args:
            sess:
            stock_count (int): num. of stocks we are going to train with.
            lstm_size (int)
            num_layers (int): num. of LSTM cell layers.
            num_steps (int)
            input_size (int)
            keep_prob (int): (1.0 - dropout rate.) for a LSTM cell.
            embed_size (int): length of embedding vector, only used when stock_count > 1.
            checkpoint_dir (str)
        """
        self.sess = sess
        self.stock_count = stock_count

        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.input_size = input_size

        self.use_embed = (embed_size is not None) and (embed_size > 0)
        self.embed_size = embed_size or -1

        self.logs_dir = logs_dir
        self.plots_dir = plots_dir

        self.build_graph()
