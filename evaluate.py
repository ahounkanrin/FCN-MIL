"""Evaluation script for the DeepLab-LargeFOV network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on around 1500 validation images.
"""

from __future__ import print_function

import argparse
#from datetime import datetime
import os
#import sys
#import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from PIL import Image

import tensorflow as tf
import numpy as np

from fcn_vgg import FCN8VGG, ImageReader, decode_labels

DATA_DIRECTORY = '/scratch/hnkmah001/Datasets/VOCdevkit'
DATA_LIST_PATH = None
VAL_DATA_LIST_PATH = './dataset-voc2011/val.txt'
NUM_STEPS = 1111
RESTORE_FROM = '/scratch/hnkmah001/FCN-MIL-GMP-LOSS/snapshots/model.ckpt-7500'
SAVE_DIR = '/scratch/hnkmah001/FCN-MIL-GMP-LOSS/val_images-ckpt-7500/'
WEIGHTS_PATH   = '/scratch/hnkmah001/Pretrained_models/vgg16.npy'
num_classes = 21

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--val_data_list", type=str, default=VAL_DATA_LIST_PATH,
                        help="Path to the file listing the images in the validation set.")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted masks.")
    parser.add_argument("--weights_path", type=str, default=WEIGHTS_PATH,
                        help="Path to the file with caffemodel weights. "
                             "If not set, all the variables are initialised randomly.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            args.val_data_list,
            input_size=None,
            random_scale=False,
            coord=coord)
        image, label = reader.val_image, reader.val_label
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add the batch dimension.
    
        # Create network.
    net = FCN8VGG(args.weights_path)


    # Predictions.
    pred = net.preds(image_batch)
    seg = pred
    pred = tf.reshape(pred, [-1,])
    gt = tf.reshape(label_batch, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(gt, num_classes - 1)), 1)  # ignore all labels >= num_classes
    gt = tf.cast(tf.gather(gt, indices), tf.int32)
    pred = tf.gather(pred, indices)   # Ignore all predictions for labels>=num_classes
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=num_classes)
    
    
     # Which variables to load.
    #trainable = tf.trainable_variables()
    all_trainable_variables = tf.trainable_variables()
    trainable = [var for var in all_trainable_variables if  'score' in var.name]
    
    print("TRAINABLE VARIABLES", trainable)



    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.initialize_all_variables()
    
    sess.run(init)
    sess.run(tf.initialize_local_variables())
    

    # Load weights.
    saver = tf.train.Saver(var_list=trainable)
    if args.restore_from is not None:
        load(saver, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    
    
    if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)

    # Iterate over images.
    for step in range(args.num_steps):
        
        image, label, segs, preds, _ = sess.run([image_batch, label_batch, seg, pred, update_op])
        
        if args.save_dir is not None:
            
#            img = decode_labels(segs[0, :, :, 0])
#            im = Image.fromarray(img)
#            im.save(args.save_dir + str(step) + '.png')
            fig, axes = plt.subplots(1, 3, figsize = (16, 12))
            
            axes.flat[0].set_title('data')
            axes.flat[0].imshow((image[0] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

            axes.flat[1].set_title('mask')
            axes.flat[1].imshow(decode_labels(label[0, :, :, 0]))

            axes.flat[2].set_title('pred')
            axes.flat[2].imshow(decode_labels(segs[0, :, :, 0]))
            plt.savefig(args.save_dir + str(step) + ".png")
            plt.close(fig)
        if step % 100 == 0:
            print('step {:d} \t'.format(step))
    print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
