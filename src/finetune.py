"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""
from __future__ import division
import csv
import numpy as np
import tensorflow as tf

from datagenerator import ImageDataGenerator
from datetime import datetime
import utils
"""
Configuration Part.
"""
class finetune(object):
    def __init__(self):
        return

    def test(self,
             meta_file="a/b/c.ckpt.meta",
             checkpoint_file="a/b/c.ckpt",
             file_list=[],
             batch_size=1,
             num_classes=16,
             device="gpu",
             dropout_rate=[1,1,1,1,1,1,1,1,1,1,1,1,1],
             training_flag=False,
             mean_pixels=[127,127,127],
             results_csv="/results.csv"):

        print("mean pixels:", mean_pixels)
        class_dict = utils.get_class_dictionary()
        with self.restore_weights(target="",
                                  config=tf.ConfigProto(allow_soft_placement=True,
                                                        log_device_placement=True),
                                  meta_file=meta_file,
                                  checkpoint_file=checkpoint_file) as sess:
            with tf.device("/"+device+":0"):
                test_data = ImageDataGenerator(file_list,
                                              mode='inference',
                                              batch_size=batch_size,
                                              num_classes=num_classes,
                                              shuffle=False,
                                               mean_pixels=mean_pixels)

                # create an reinitializable iterator given the dataset structure
                iterator = tf.data.Iterator.from_structure(test_data.data.output_types,
                                                   test_data.data.output_shapes)
                next_batch = iterator.get_next()

            # Ops for initializing the two different iterators
            testing_init_op = iterator.make_initializer(test_data.data)

            # Initialize model
            graph = tf.get_default_graph()

            # TF placeholder for graph input and output
            x = graph.get_tensor_by_name("x:0")#("Placeholder:0")
            y = graph.get_tensor_by_name("y:0")#("Placeholder_1:0")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")#("Placeholder_2:0")
            train_flag = graph.get_tensor_by_name("training_flag:0")#("Placeholder_3:0")

            #print tensor name
            tensors=[n.name for n in graph.as_graph_def().node]
            np_tensor=np.array(tensors)
            np.savetxt("mynet_tensors.txt",np_tensor,fmt="%s")
            ops=[o.name for o in tf.get_default_graph().get_operations()]
            np_tensor=np.array(ops)
            np.savetxt("mynet_op_names.txt",np_tensor,fmt="%s")

            score=graph.get_tensor_by_name("fc8/fc8:0")
            predict_op=tf.arg_max(score,1)
            GT_op = tf.arg_max(y, 1)
            correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))

            # Get the number of training/validation steps per epoch
            test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))

            # Validate the model on the entire validation set
            print("{} Start TESTING..".format(datetime.now()))
            sess.run(testing_init_op)
            test_acc = 0.
            test_count = 0
            correct_count=0
            all_img_test_count=0
            predict_arr=[]
            gt_arr=[]
            matches=[]
            for _ in range(test_batches_per_epoch):
                try:
                    img_batch, label_batch = sess.run(next_batch)
                    preds = sess.run(predict_op, feed_dict={x: img_batch,
                                                            y: label_batch,
                                                            keep_prob: dropout_rate,
                                                            train_flag:training_flag})
                    predict_arr += [c for c in preds]

                    gt = sess.run(GT_op, feed_dict={x: img_batch,
                                                    y: label_batch,
                                                    keep_prob: dropout_rate,
                                                            train_flag:training_flag})
                    gt_arr += [c for c in gt]
                    matched=[i for i, j in zip(preds, gt) if i == j]
                    correct_count += len(matched)

                    all_img_test_count += img_batch.shape[0]
                    # print "PREDICTED: %s, GT: %s" % ( predict_arr, gt_arr)
                    print "PREDICTED: %s" %([class_dict[p] for p in predict_arr])
                    
                except Exception,e:
                    print "EXCEPTION:%s"%e
                    continue
            print("{} Testing Accuracy = {:.4f}".format(datetime.now(),
                                                        1))

            #sess.close()
            results = zip(gt_arr, predict_arr)
            np.savetxt(results_csv, results, fmt="%s", delimiter=" ")
            np_predict=np.array(predict_arr)
            np_gt=np.array(gt_arr)
            

    def restore_weights(self,
                        target="",
                        config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True),
                        meta_file="a/b/c.ckpt.meta",
                        checkpoint_file="a/b/c.ckpt"):

        sess=tf.Session(target=target,config=config)
        #saver = tf.train.Saver()
        saver=tf.train.import_meta_graph(meta_file)
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,checkpoint_file)
        return sess

