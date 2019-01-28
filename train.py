"""Training script for the DeepLab-LargeFOV network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC dataset,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
#from datetime import datetime
import os
#import sys
import time
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

#tf.enable_eager_execution()


from fcn_vgg import FCN8VGG, ImageReader, decode_labels

BATCH_SIZE = 20
DATA_DIRECTORY = '/scratch/hnkmah001/Datasets/VOCdevkit'
DATA_LIST_PATH = './dataset/train.txt'
VAL_DATA_LIST_PATH = './dataset/val.txt'
INPUT_SIZE = '321,321'
#INPUT_SIZE = None
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
MEAN_IMG = tf.Variable(np.array((104.00698793,116.66876762,122.67891434)), trainable=False, dtype=tf.float32)
NUM_STEPS = 10000
VALIDATION_STEPS = 1449//BATCH_SIZE
RANDOM_SCALE = True
#RESTORE_FROM = './snapshots/model.ckpt-2000'
RESTORE_FROM = None
SAVE_DIR = './images/'
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 500
SNAPSHOT_DIR = './snapshots/'
WEIGHTS_PATH   = '/scratch/hnkmah001/Pretrained_models/vgg16.npy'
SUMMARIES_DIR  = './summaries/'

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="MIL-FCN8VGG16 Network")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the training set.")
    parser.add_argument("--val_data_list", type=str, default=VAL_DATA_LIST_PATH,
                        help="Path to the file listing the images in the validation set.")
    parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum parameter")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                        help="Weight decay parameter")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--val_steps", type=int, default=VALIDATION_STEPS,
                        help="Number of validation steps.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save figures with predictions.")
    parser.add_argument("--save_num_images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
                        help="Save figure with predictions and ground truth every often.")
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weights_path", type=str, default=WEIGHTS_PATH,
                        help="Path to the file with caffemodel weights. "
                             "If not set, all the variables are initialised randomly.")
    parser.add_argument("--summaries_dir", type=str, default= SUMMARIES_DIR,
                        help="Path to the file where variables are saved for TensorBoard.")
    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')
    
def load(loader, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      loader: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    loader.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the training."""
    args = get_arguments()
    
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            args.val_data_list,
            input_size,
            RANDOM_SCALE,
            coord)
        train_batch = reader.dequeue_train(args.batch_size)
        image_batch, label_batch = train_batch
        val_data = reader.dequeue_val(args.batch_size)

    # Create network  with weights initialized from vgg16 pretrained on ImageNet
    net = FCN8VGG(args.weights_path)

    # Define the loss and optimisation parameters.
    
    loss = net.mil_loss(train_batch, is_training=False)
#    optimiser = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
#    optimiser = tf.train.MomentumOptimizer(learning_rate=args.learning_rate, momentum=args.momentum)
    optimiser = tf.contrib.opt.MomentumWOptimizer(weight_decay=args.weight_decay, learning_rate=args.learning_rate, momentum=args.momentum)
    all_trainable_variables = tf.trainable_variables()
    trainable = [var for var in all_trainable_variables if  'score' in var.name]
    optim = optimiser.minimize(loss, var_list=trainable)
    
#    print("\n\nTHE WEIGHT DEDCAY LOSS:" , net.mil_loss.loss.weight_decay_loss)
    
    val_loss = net.mil_loss(val_data)
    pred = net.preds(image_batch)
    
    
    val_loss_glob = tf.Variable(0.0, trainable=False)
    val_loss_i_value = tf.Variable(0.0, trainable=False)
    
    loss_summary = tf.summary.scalar('loss', loss)
    
#    val_loss_summary = tf.summary.scalar('validation loss', val_loss_glob)
#    merged = tf.summary.merge_all()
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    
    if os.path.exists(args.summaries_dir):
        shutil.rmtree(args.summaries_dir)
        
    train_writer = tf.summary.FileWriter(args.summaries_dir + '/train',
                                      sess.graph)
    val_writer = tf.summary.FileWriter(args.summaries_dir + '/val')
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    
    print("TRAINABLE VARIABLES: ", trainable)
   
    # Saver for storing the last 10 checkpoints of the model.
    saver = tf.train.Saver(var_list=trainable, max_to_keep=10)
    if args.restore_from is not None:
        load(saver, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
   
    # Iterate over training steps.
    starting_time=time.asctime(time.localtime())
    with open('timelogs.txt', 'w+') as f:  
        print("Training started on: ", starting_time, file=f)
    
    for step in range(args.num_steps+1):
        start_time = time.time()
       
        
        if step % args.save_pred_every == 0:
            
            # Calculate the validation loss over the whole validation set
            val_loss_ = tf.Variable(0.0, trainable=False)
            sess.run(val_loss_.initializer)
            for i in range(args.val_steps): 
                val_loss_i_value = sess.run(val_loss)
#                print("Validation loss of {:d}th batch is {:.3f}".format(i, val_loss_i_value))
                val_loss_ = val_loss_ + val_loss_i_value
            
            
            val_loss_glob = tf.divide(val_loss_ , args.val_steps) # validation loss of the whole validation data
            
            summary=tf.Summary()
            summary.value.add(tag='validation loss', simple_value = sess.run(val_loss_glob))
            
#            val_loss_summary = tf.summary.scalar('validation loss', val_loss_glob)
                
            val_loss_value, loss_value, images, labels, preds, _ = sess.run([ val_loss_glob, loss, image_batch, label_batch, pred, optim])
            fig, axes = plt.subplots(args.save_num_images, 3, figsize = (16, 12))
            for i in range(args.save_num_images):
                axes.flat[i * 3].set_title('data')
                axes.flat[i * 3].imshow((images[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

                axes.flat[i * 3 + 1].set_title('mask')
                axes.flat[i * 3 + 1].imshow(decode_labels(labels[i, :, :, 0]))

                axes.flat[i * 3 + 2].set_title('pred')
                axes.flat[i * 3 + 2].imshow(decode_labels(preds[i, :, :, 0]))
            plt.savefig(args.save_dir + str(start_time) + ".png")
            plt.close(fig)
            save(saver, sess, args.snapshot_dir, step)
            val_writer.add_summary(summary, step)

            duration = time.time() - start_time
            print("\nSTEP {:d}/{:d} VALIDATION LOSS = {:.8f}, \t ({:.3f} sec/step)".format(step, args.num_steps, val_loss_value, duration))
            #print('step {:d} \t loss = {:.8f}, \t ({:.3f} sec/step)'.format(step, loss_value, duration))
		
        else:
            summary, loss_value, _ = sess.run([loss_summary, loss, optim])
            train_writer.add_summary(summary, step)
            duration = time.time() - start_time
            print('step {:d}/{:d} \t loss = {:.8f}, \t ({:.3f} sec/step)'.format(step, args.num_steps, loss_value, duration))
    end_time=time.asctime(time.localtime())
    with open('timelogs.txt', 'a') as f:  
        print("Training ended on: ", end_time, file=f)               
		      
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
