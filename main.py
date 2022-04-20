
from datetime import datetime
import numpy as np
import optparse
import os
import sys
import tensorflow as tf

import config
from ds import loader
from models.model import DeepIST



def main(training_fname, options):
    
    if os.path.exists(config.validation_log):
        os.remove(config.validation_log)

    assert config.max_len is not None, "Please set 'max_len' in config.py based on your data"
    assert config.image_folder is not None, "Please set 'image_folder path file' in config.py"

    training_loader = loader.DataGenerator(training_fname,
                                           config.image_folder,
                                           config.batch_size,
                                           config.image_width,
                                           config.image_height,
                                           max_len=config.max_len,
                                           channel=config.channel,
                                           ratio=config.ratio)
    train(training_loader)
    return 0


def train(training_loader):
    x = tf.placeholder(tf.float32,
                       [config.batch_size,
                        config.max_len,
                        config.image_width,
                        config.image_height,
                        config.channel])
    lengths = tf.placeholder(tf.float32, [config.batch_size])
    y = tf.placeholder(tf.float32, [config.batch_size, 1])
    z = tf.placeholder(tf.float32, [None, 1])

    model = DeepIST(image_input=x,
                    length_input=lengths,
                    max_len=config.max_len,
                    channel=config.channel,
                    cnn1d_dim=config.cnn1d_dim, cnn2d_dim=config.cnn2d_dim,
                    cnn2d_convs=config.cnn2d_convs,
                    filter_size=config.filter_size)
    sub_predictions = model.sub_final
    predictions = model.final
    parameters = tf.trainable_variables()

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.divide(tf.abs(tf.divide(tf.subtract(predictions,y),(tf.math.abs(y) + tf.math.abs(predictions) + 1e-10))),2))
        sub_loss = tf.reduce_mean(tf.divide(tf.abs(tf.divide(tf.subtract(sub_predictions,y),(tf.math.abs(y) + tf.math.abs(sub_predictions) + 1e-10))),2))
        total_loss = loss+sub_loss
    
        penalty = 0
    #    total_loss = loss+sub_loss

    with tf.name_scope('optimizer'):
        train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(total_loss)

    tf.summary.scalar('loss', loss)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(config.filewriter_path)
    saver = tf.train.Saver()

    step = 100
    valid_step = 1000
    valid_count = 1000
    save_step = 10000
    losses, sub_losses, penalties = [], [], []
    test_mse = []
    test_mae = []
    test_mape = []

    test_smape = [] #List for comparision purposes
    
    print (config.cnn2d_convs)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=sess)
        writer.add_graph(sess.graph)

        for iteration in range(config.num_iter):
            image_batch, length_batch, sub_value_batch, value_batch = sess.run(tf.data(
                    [training_loader.train_image_batch,
                     training_loader.train_lenght_batch,
                     training_loader.train_sub_value_batch,
                     training_loader.train_value_batch]))
            sub_values = []
            for length, sub_value_list in zip(length_batch, sub_value_batch):
                sub_values.extend(sub_value_list[:length])
            _, l, sub_l, sub_p = sess.run([train_op, loss, sub_loss, sub_predictions],
                                           feed_dict={x: image_batch,
                                                      lengths: length_batch,
                                                      y: value_batch,
                                                      z: sub_values})


            losses.append(l)
            sub_losses.append(sub_l)

            if iteration % step == 0:
                print("{} Iteration {} loss: {:.3f} sub_loss: {:.3f}, penalties: {:.3f}".format(
                    datetime.now(),
                    iteration,
                    sum(losses)/step,
                    sum(sub_losses)/step,
                    sum(penalties)/step))
                losses, sub_losses, penalties = [], [], []


            if iteration > 0 and iteration % valid_step == 0:
                for _ in range(valid_count):
                    image_batch, length_batch, sub_value_batch, value_batch = sess.run(tf.data([
                        training_loader.test_image_batch,
                        training_loader.test_length_batch,
                        training_loader.test_sub_value_batch,
                        training_loader.test_value_batch]))
                    sub_values = []
                    for length, sub_value_list in zip(length_batch, sub_value_batch):
                        sub_values.extend(sub_value_list[:length])
                    predicted, sub_predicted = sess.run(
                        [predictions, sub_predictions],
                        feed_dict={x: image_batch,
                                   lengths: length_batch,
                                   y: value_batch,
                                   z: sub_values})
                    mse = np.mean(np.square(predicted - value_batch))
                    mae = np.mean(np.abs(predicted - value_batch))
                    mape = np.mean(np.abs(predicted - value_batch) / value_batch)
                    smape = np.mean(np.abs(predicted - value_batch) / (value_batch + predicted))/2
                    test_mse.append(mse)
                    test_mae.append(mae)
                    test_mape.append(mape)
                    test_smape.append(smape)
                    
                print("{} Validation MSE: {:.3f}, MAE: {:.3f}, MAPE: {:.3f}".format(
                    datetime.now(),
                    sum(test_mse)/len(test_mse),
                    sum(test_mae)/len(test_mae),
                    sum(test_mape)/len(test_mape),
                    sum(test_smape)/len(test_smape)))
                test_mse = []
                test_mae = []
                test_mape = []
                test_smape = []

            if iteration != 0 and iteration % save_step == 0:
                print("{} Saving checkpoint of model...".format(datetime.now()))
                checkpoint_name = os.path.join(config.checkpoint_path, 'iter' + str(iteration) + '.ckpt')
                save_path = saver.save(sess, checkpoint_name)


if __name__ == '__main__':
    
        
    parser = optparse.OptionParser(usage=main.__doc__)
    options, args = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
        sys.exit()

    sys.exit(main(args[0], options))

