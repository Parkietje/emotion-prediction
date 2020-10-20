from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import data_process
import numpy as np
import os


slim = tf.contrib.slim


# Create FLAGS
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 1, '''The batch size to use.''')
tf.app.flags.DEFINE_integer('seq_length', 1, 'the sequence length: how many consecutive frames to use for the RNN; if the network is only CNN then put here any number you want : total_batch_size = batch_size * seq_length')
tf.app.flags.DEFINE_integer('size', 96, 'dimensions of input images, e.g. 96x96')
tf.app.flags.DEFINE_string('network',  'affwildnet_vggface' , ' which network architecture we want to use,  pick between : vggface_4096, vggface_2000, affwildnet_vggface, affwildnet_resnet '     )                           
tf.app.flags.DEFINE_string('input_file',  os.path.join(os.getcwd(), 'input.csv') , 'the input file : it should be in the format: image_file_location,valence_value,arousal_value  and images should be jpgs'     )                           
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', os.path.join(os.getcwd(), 'pretrained_models/vggface_rnn/model.ckpt-0'), 'the pretrained model checkpoint path to restore,if there exists one  ')


###############################################################################################################################################################
####  The sample code and the model weights are for RESEARCH PURPOSES only and cannot be used for commercial use.      ########################################
####                                 Do not redistribute this elsewhere                                                ########################################
################################################################################################################################################################

def predict(image_path):
  g = tf.Graph()
  with g.as_default():

    # overwrite input file with path of image to predict
    input_file = FLAGS.input_file
    with open(input_file, 'w') as f:
      f.write(image_path+',0,0')

    #read input data
    image_list, label_list = data_process.read_labeled_image_list(input_file)
    # split into sequences, note: in the cnn models case this is splitting into batches of length: seq_length ;
    # for the cnn-rnn models case, I do not check whether the images in a sequence are consecutive or the images are from the same video/the images are displaying the same person 
    image_list, label_list = data_process.make_rnn_input_per_seq_length_size(image_list,label_list,FLAGS.seq_length)

    images = tf.convert_to_tensor(image_list)
    labels = tf.convert_to_tensor(label_list)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels,images],num_epochs=None, shuffle=False, seed=None,capacity=1000, shared_name=None, name=None)
    images_batch, labels_batch, image_locations_batch = data_process.decodeRGB(input_queue,FLAGS.seq_length,FLAGS.size)
    images_batch = tf.to_float(images_batch)
    images_batch -= 128.0
    images_batch /= 128.0  # scale all pixel values in range: [-1,1]

    images_batch = tf.reshape(images_batch,[-1,96,96,3])
    labels_batch = tf.reshape(labels_batch,[-1,2])

    if FLAGS.network == 'affwildnet_vggface':
     from affwildnet import vggface_gru as net
     network = net.VGGFace(FLAGS.batch_size, FLAGS.seq_length)
     network.setup(images_batch)
     prediction = network.get_output()
    
    #
    num_batches = int(len(image_list)/FLAGS.batch_size)
    variables_to_restore =  tf.global_variables()
    
    #
    with tf.Session() as sess:
      init_fn = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_checkpoint_path, variables_to_restore,ignore_missing_vars=False)
      init_fn(sess)
      print('Loading model {}'.format(FLAGS.pretrained_model_checkpoint_path))
      tf.train.start_queue_runners(sess=sess)
      coord = tf.train.Coordinator()
      evaluated_predictions = []
      evaluated_labels = []
      images = []
      
      try:
        for _ in range(num_batches):
          pr, l,imm = sess.run([prediction,labels_batch, image_locations_batch])
          evaluated_predictions.append(pr)
          evaluated_labels.append(l)
          images.append(imm)
          if coord.should_stop():
            break
        coord.request_stop()
      except Exception as e:
        coord.request_stop(e)
      
      predictions = np.reshape(evaluated_predictions, (-1, 2))
      labels = np.reshape(evaluated_labels, (-1, 2))
      images = np.reshape(images, (-1))

      valence = sum((predictions[:,0]))/len(predictions[:,0])
      print('Valence : {}'.format(valence))
      arousal = sum((predictions[:,1]))/len(predictions[:,1])
      print('Arousal : {}'.format(arousal))
  
  return valence, arousal


if __name__ == '__main__':
    from sys import argv
    
    # use from command line: python predict.py YOUR_INPUT_FILE
    if len(argv) == 2:
      predict(str(argv[1]))

    # use default input file
    else:
      predict('test_data/happy.jpg')