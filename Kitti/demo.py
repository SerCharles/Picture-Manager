"""
Detects Cars in an image using KittiSeg.

Input: Image
Output: Image (with Cars plotted in Green)

Utilizes: Trained KittiSeg weights. If no logdir is given,
pretrained weights will be downloaded and used.

Usage:
python demo.py --input_image data/demo.png [--output_image output_image]
                [--logdir /path/to/weights] [--gpus 0]

--------------------------------------------------------------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann

Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys
import imp

import collections
import matplotlib.cm as cm

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf

sess = None
prediction = None
hypes = None
image_pl = None


flags = tf.app.flags
FLAGS = flags.FLAGS

def make_overlay(image, gt_prob):

    mycm = cm.get_cmap('bwr')

    overimage = mycm(gt_prob, bytes=True)
    output = 0.4*overimage[:,:,0:3] + 0.6*image

    return output


flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('input_image', None,
                    'Image to apply KittiSeg.')
flags.DEFINE_string('output_image', None,
                    'Image to apply KittiSeg.')


default_run = 'KittiSeg_pretrained'
weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/KittiSeg_pretrained.zip")


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image

def _add_paths_to_sys(hypes):
    """
    Add all module dirs to syspath.

    This adds the dirname of all modules to path.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    """
    base_path = hypes['dirs']['base_path']
    if 'path' in hypes:
            for path in hypes['path']:
                path = os.path.realpath(os.path.join(base_path, path))
                sys.path.insert(1, path)
    return

def load_hypes_from_logdir(logdir, subdir="model_files", base_path=None):
    """Load hypes from the logdir.

    Namely the modules loaded are:
    input_file, architecture_file, objective_file, optimizer_file

    Parameters
    ----------
    logdir : string
        Path to logdir

    Returns
    -------
    hypes
    """
    model_dir = os.path.join(logdir, subdir)
    hypes_fname = os.path.join(model_dir, "hypes.json")
    with open(hypes_fname, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)

    hypes['dirs']['output_dir'] = os.path.realpath(logdir)
    hypes['dirs']['image_dir'] = os.path.join(hypes['dirs']['output_dir'],
                                              'images')

    if base_path is not None:
        hypes['dirs']['base_path'] = os.path.realpath(base_path)

    _add_paths_to_sys(hypes)

    if 'TV_DIR_DATA' in os.environ:
        data_dir = os.environ['TV_DIR_DATA']
    else:
        data_dir = 'DATA'

    hypes['dirs']['data_dir'] = data_dir

    return hypes


def load_modules_from_hypes(hypes, postfix=""):
    """Load all modules from the files specified in hypes.

    Namely the modules loaded are:
    input_file, architecture_file, objective_file, optimizer_file

    Parameters
    ----------
    hypes : dict
        Hyperparameters

    Returns
    -------
    hypes, data_input, arch, objective, solver
    """
    modules = {}
    base_path = hypes['dirs']['base_path']

    # _add_paths_to_sys(hypes)
    f = os.path.join(base_path, hypes['model']['input_file'])
    data_input = imp.load_source("input_%s" % postfix, f)
    modules['input'] = data_input

    f = os.path.join(base_path, hypes['model']['architecture_file'])
    arch = imp.load_source("arch_%s" % postfix, f)
    modules['arch'] = arch

    f = os.path.join(base_path, hypes['model']['objective_file'])
    objective = imp.load_source("objective_%s" % postfix, f)
    modules['objective'] = objective

    f = os.path.join(base_path, hypes['model']['optimizer_file'])
    solver = imp.load_source("solver_%s" % postfix, f)
    modules['solver'] = solver

    f = os.path.join(base_path, hypes['model']['evaluator_file'])
    eva = imp.load_source("evaluator_%s" % postfix, f)
    modules['eval'] = eva

    return modules

def build_inference_graph(hypes, modules, image):
    """Run one evaluation against the full epoch of data.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    modules : tuble
        the modules load in utils
    image : placeholder

    return:
        graph_ops
    """
    with tf.name_scope("Validation"):

        logits = modules['arch'].inference(hypes, image, train=False)

        decoded_logits = modules['objective'].decoder(hypes, logits,
                                                      train=False)
    return decoded_logits

def load_modules_from_logdir(logdir, dirname="model_files", postfix=""):
    """Load hypes from the logdir.

    Namely the modules loaded are:
    input_file, architecture_file, objective_file, optimizer_file

    Parameters
    ----------
    logdir : string
        Path to logdir

    Returns
    -------
    data_input, arch, objective, solver
    """
    model_dir = os.path.join(logdir, dirname)
    f = os.path.join(model_dir, "data_input.py")
    # TODO: create warning if file f does not exists
    data_input = imp.load_source("input_%s" % postfix, f)
    f = os.path.join(model_dir, "architecture.py")
    arch = imp.load_source("arch_%s" % postfix, f)
    f = os.path.join(model_dir, "objective.py")
    objective = imp.load_source("objective_%s" % postfix, f)
    f = os.path.join(model_dir, "solver.py")
    solver = imp.load_source("solver_%s" % postfix, f)

    f = os.path.join(model_dir, "eval.py")
    eva = imp.load_source("evaluator_%s" % postfix, f)
    modules = {}
    modules['input'] = data_input
    modules['arch'] = arch
    modules['objective'] = objective
    modules['solver'] = solver
    modules['eval'] = eva

    return modules


def setup(_):
    global sess
    global hypes
    global prediction
    global image_pl
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    if FLAGS.logdir is None:
        # Download and use weights from the MultiNet Paper
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                    'KittiSeg')
        else:
            runs_dir = 'Kitti/RUNS'
        logdir = os.path.join(runs_dir, default_run)
    else:
        logging.info("Using weights found in {}".format(FLAGS.logdir))
        logdir = FLAGS.logdir

    # Loading hyperparameters from logdir
    hypes = load_hypes_from_logdir(logdir, base_path='hypes')

    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = load_modules_from_logdir(logdir)
    logging.info("Modules loaded successfully. Starting to build tf graph.")

    # Create tf graph and build module.
    with tf.Graph().as_default():
        # Create placeholder for input
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)

        # build Tensorflow graph using the model from logdir
        prediction = build_inference_graph(hypes, modules,
                                                image=image)

        logging.info("Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        load_weights(logdir, sess, saver)

        logging.info("Weights loaded successfully.")

def load_weights(checkpoint_dir, sess, saver):
    """
    Load the weights of a model stored in saver.

    Parameters
    ----------
    checkpoint_dir : str
        The directory of checkpoints.
    sess : tf.Session
        A Session to use to restore the parameters.
    saver : tf.train.Saver

    Returns
    -----------
    int
        training step of checkpoint
    """
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        logging.info(ckpt.model_checkpoint_path)
        file = os.path.basename(ckpt.model_checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_dir, file)
        saver.restore(sess, checkpoint_path)
        return int(file.split('-')[1])

def segmentation(image):
    global prediction
    global sess
    global hypes
    global image_pl
    if hypes['jitter']['reseize_image']:
        # Resize input only, if specified in hypes
        image_height = hypes['jitter']['image_height']
        image_width = hypes['jitter']['image_width']
        image = scp.misc.imresize(image, size=(image_height, image_width),
                                  interp='cubic')
    # Run KittiSeg model on image
    feed = {image_pl: image}
    softmax = prediction['softmax']
    output = sess.run([softmax], feed_dict=feed)
    # Reshape output from flat vector to 2D Image
    shape = image.shape
    output_image = output[0][:, 1].reshape(shape[0], shape[1])
    rb_image = make_overlay(image, output_image)
    return rb_image


# def setup():
#     tf.app.run()
