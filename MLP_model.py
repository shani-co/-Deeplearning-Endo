 #!/usr/bin/env python

import tensorflow._api.v2.compat.v1 as tf, numpy as np, pandas
from datetime import datetime as dt
from os.path import join

# import tensorflow as tf
tf.disable_v2_behavior()

import numpy as np
import pandas
from datetime import datetime as dt

constants = {
    'alpha': 0.00004,
    'epochs': 100000,
    'features': 58,
    'labels': 2,
    'bin_size': 300,
    'bins': None,
    'hidden1_size':15,
    'hidden2_size':30,
}
#Our beloved features
wanted_columns= ['Heavy / Extreme menstrual bleeding',
                 'Menstrual pain (Dysmenorrhea)',
                 'Painful / Burning pain during sex (Dyspareunia)',
                 'Pelvic pain',
                 'Irregular / Missed periods',
                 'Cramping',
                 'Abdominal pain / pressure',
                 'Back pain',
                 'Painful bowel movements',
                 'Nausea',
                 'Menstrual clots',
                 'Infertility',
                 'Painful cramps during period',
                 'Pain / Chronic pain',
                 'Diarrhea',
                 'Long menstruation',
                 'Constipation / Chronic constipation',
                 'Vomiting / constant vomiting',
                 'Fatigue / Chronic fatigue',
                 'Painful ovulation',
                 'Stomach cramping',
                 'Migraines',
                 'Extreme / Severe pain',
                 'Leg pain',
                 'Irritable Bowel Syndrome (IBS)',
                 'Syncope (fainting, passing out)',
                 'Mood swings',
                 'Depression',
                 'Bleeding',
                 'Lower back pain',
                 'Fertility Issues',
                 'Ovarian cysts',
                 'Painful urination',
                 'Headaches',
                 'Constant bleeding',
                 'Pain after Intercourse',
                 'Digestive / GI problems',
                 'IBS-like symptoms',
                 'Excessive bleeding',
                 'Anaemia / Iron deficiency',
                 'Hip pain',
                 'Vaginal Pain/Pressure',
                 'Sharp / Stabbing pain',
                 'Bowel pain',
                 'Anxiety',
                 'Cysts (unspecified)',
                 'Dizziness',
                 'Malaise / Sickness',
                 'Abnormal uterine bleeding',
                 'Fever',
                 'Hormonal problems',
                 'Bloating',
                 'Feeling sick',
                 'Decreased energy / Exhaustion',
                 'Abdominal Cramps during Intercourse',
                 'Insomnia / Sleeplessness',
                 'Acne / pimples',
                 'Loss of appetite',
                 'label'
                 ]



def shuffle_df(raw_data):
    return raw_data.sample(frac=1).reset_index(drop=True)


def get_features_and_labels(data):
    dataset_labels = data.loc[:,'label']
    dataset_features = data.drop(columns=['label'])
    return dataset_features, dataset_labels


def get_shuffled_divided_data(raw_data):
    data = shuffle_df(raw_data)
    #data = fit_labels(data)
    df, labels = get_features_and_labels(data)
    TRAIN_SIZE = int(len(data) * 0.6)
    train_df = df[:TRAIN_SIZE]
    train_labels = labels[:TRAIN_SIZE]

    test_df = df[TRAIN_SIZE:]
    test_labels = labels[TRAIN_SIZE:]

    return train_df, train_labels, test_df, test_labels


def main():
    data_path = join("/home/cyberlab/Desktop/Shani's_ML/Deeplearning-Endo",'Dataset .xlsx')
    raw_data = pandas.read_excel(data_path)
    raw_data = raw_data[wanted_columns]
    x_o_train, y_o_train, x_o_test, y_o_test = get_shuffled_divided_data(raw_data)

    # initialize input and output vectors
    X = tf.placeholder(tf.float32, [None, constants['features']])
    Y = tf.placeholder(tf.float32, [None, constants['labels']])

    # initialize weights and biases randomly

    ###################################################################################################################
    W1 = tf.Variable(0.001 * np.random.randn(constants['features'], constants['hidden1_size']).astype(np.float32))
    B1 = tf.Variable(0.001 * np.random.randn(constants['hidden1_size']).astype(np.float32))
    Z1 = tf.nn.relu(tf.matmul(X, W1) + B1)
    W2 = tf.Variable(0.001 * np.random.randn(constants['hidden1_size'], constants['hidden2_size']).astype(np.float32))
    B2 = tf.Variable(0.001 * np.random.randn(constants['hidden2_size']).astype(np.float32))
    Z2 = tf.nn.relu(tf.matmul(Z1, W2) + B2)
    W3 = tf.Variable(0.001 * np.random.randn(constants['hidden2_size'], constants['labels']).astype(np.float32))
    B3 = tf.Variable(0.001 * np.random.randn(constants['labels']).astype(np.float32))
    ###################################################################################################################

    Y_ = tf.nn.softmax(tf.add(tf.matmul(Z2, W3), B3))

    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(tf.clip_by_value(Y_, 1e-10, 1.0))))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=constants['alpha']).minimize(cost)

    init = tf.global_variables_initializer()

    # calculate bin size for training
    constants['bins'] = int(x_o_train.shape[0] / constants['bin_size'])

    training_start_time = dt.now()

    # start tensor session
    with tf.Session() as sess:
        y_o_train = sess.run(tf.one_hot(y_o_train, constants['labels']))
        y_o_test = sess.run(tf.one_hot(y_o_test, constants['labels']))

        sess.run(init)

        cost_hist = []

        # model training (optimizer)
        for epoch in range(constants['epochs']):
            for bi in range(constants['bins']):

                start_point = bi * epoch
                end_point = start_point + constants['bin_size']

                x = x_o_train[start_point: end_point]
                y = y_o_train[start_point: end_point]

                sess.run(optimizer, feed_dict={X: x, Y: y})
                c = sess.run(cost, feed_dict={X: x, Y: y})

                if (epoch % 500 == 0 and epoch != 0) or (epoch == constants['epochs'] - 1):
                    cost_hist.append(c)
                    print('\rEpoch: {} Cost: {}'.format(str(epoch), str(c)))

        training_end_time = dt.now()

        # model testing
        correct_prediction = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = accuracy.eval({X: x_o_test, Y: y_o_test}) * 100

        # ready the stat output
        first_cost = cost_hist[0]
        last_cost = cost_hist[-1]
        avg_cost = sum(cost_hist) / len(cost_hist)
        cost_hist.sort()
        lowest_cost = cost_hist[0]
        highest_cost = cost_hist[-1]
        training_time = training_end_time - training_start_time

        print('Finished Running\n')

        print('Running Details:\n'
              '\tNumber of Epochs Set To : {}\n'.format(constants['epochs']) +
              '\tNumber of Bins Set To : {}\n'.format(constants['bins'] + 1) +
              '\tSize of Bin Per Epoch : {}\n'.format(constants['bin_size']) +
              '\tTotal Training Cycles : {}\n'.format(constants['epochs'] * (constants['bins']) + 1) +
              '\tLearning Rate Set To : {}\n'.format(constants['alpha']) +
              '\tTotal Training Time : {}\n'.format(str(training_time))
              )

        print('Costs:\n'
              '\tFirst Recorded Cost : {}\n'.format(first_cost) +
              '\tLast Recorded Cost : {}\n'.format(last_cost) +
              '\tAverage Cost : {}\n'.format(avg_cost) +
              '\tLowest Recorded Cost : {}\n'.format(lowest_cost) +
              '\tHighest Recorded Cost : {}\n'.format(highest_cost)
              )

        print('Accuracy:\n'
              '\tFinal Accuracy: {} %\n'.format(acc)
              )

        print('Confusion Matrix:')

        # confusion matrix
        conf_mat = tf.confusion_matrix(labels=tf.argmax(Y, 1), predictions=tf.argmax(Y_, 1), num_classes=2)
        conf_mat_to_print = sess.run(conf_mat, feed_dict={X: x_o_test, Y: y_o_test})
        recall = str((conf_mat_to_print[0,0] /(conf_mat_to_print[0,0]+conf_mat_to_print[0,1]))*100)
        precision = str((conf_mat_to_print[0,0] /(conf_mat_to_print[0,0]+conf_mat_to_print[1,0]))*100)
        print("Final Recall:"+recall+"%\n")
        print("Final Precision:"+precision+"%\n")
        print(conf_mat_to_print)


if _name_ == '_main_':

    main()
import tensorflow._api.v2.compat.v1 as tf, numpy as np, pandas
from datetime import datetime as dt
from os.path import join

tf.disable_v2_behavior()

constants = {
    'features': 58,
    'labels': 2,
    'alpha': 0.0001,
    'epochs': 60000,
    'bin_size': 300,
    'bins': None
}
#Our beloved features
wanted_columns= ['Heavy / Extreme menstrual bleeding',
                 'Menstrual pain (Dysmenorrhea)',
                 'Painful / Burning pain during sex (Dyspareunia)',
                 'Pelvic pain',
                 'Irregular / Missed periods',
                 'Cramping',
                 'Abdominal pain / pressure',
                 'Back pain',
                 'Painful bowel movements',
                 'Nausea',
                 'Menstrual clots',
                 'Infertility',
                 'Painful cramps during period',
                 'Pain / Chronic pain',
                 'Diarrhea',
                 'Long menstruation',
                 'Constipation / Chronic constipation',
                 'Vomiting / constant vomiting',
                 'Fatigue / Chronic fatigue',
                 'Painful ovulation',
                 'Stomach cramping',
                 'Migraines',
                 'Extreme / Severe pain',
                 'Leg pain',
                 'Irritable Bowel Syndrome (IBS)',
                 'Syncope (fainting, passing out)',
                 'Mood swings',
                 'Depression',
                 'Bleeding',
                 'Lower back pain',
                 'Fertility Issues',
                 'Ovarian cysts',
                 'Painful urination',
                 'Headaches',
                 'Constant bleeding',
                 'Pain after Intercourse',
                 'Digestive / GI problems',
                 'IBS-like symptoms',
                 'Excessive bleeding',
                 'Anaemia / Iron deficiency',
                 'Hip pain',
                 'Vaginal Pain/Pressure',
                 'Sharp / Stabbing pain',
                 'Bowel pain',
                 'Anxiety',
                 'Cysts (unspecified)',
                 'Dizziness',
                 'Malaise / Sickness',
                 'Abnormal uterine bleeding',
                 'Fever',
                 'Hormonal problems',
                 'Bloating',
                 'Feeling sick',
                 'Decreased energy / Exhaustion',
                 'Abdominal Cramps during Intercourse',
                 'Insomnia / Sleeplessness',
                 'Acne / pimples',
                 'Loss of appetite',
                 'label'
                 ]


def print_stats(acc, cost_hist, training_end_time, training_start_time):
    # ready the stat output
    first_cost = cost_hist[0]
    last_cost = cost_hist[-1]
    avg_cost = sum(cost_hist) / len(cost_hist)
    cost_hist.sort()
    lowest_cost = cost_hist[0]
    highest_cost = cost_hist[-1]
    training_time = training_end_time - training_start_time
    print('Finished Running\n')
    print('\tRunning Details:\n'
          '\t\tNumber of Epochs Set To : {}\n'.format(constants['epochs']) +
          '\t\tNumber of Bins Set To : {}\n'.format(constants['bins'] + 1) +
          '\t\tSize of Bin Per Epoch : {}\n'.format(constants['bin_size']) +
          '\t\tTotal Training Cycles : {}\n'.format(constants['epochs'] * (constants['bins'] + 1)) +
          '\t\tLearning Rate Set To : {}\n'.format(constants['alpha']) +
          '\t\tTotal Training Time : {}\n'.format(str(training_time))
          )
    print('\tCosts:\n'
          '\t\tFirst Recorded Cost : {}\n'.format(first_cost) +
          '\t\tLast Recorded Cost : {}\n'.format(last_cost) +
          '\t\tAverage Cost : {}\n'.format(avg_cost) +
          '\t\tLowest Recorded Cost : {}\n'.format(lowest_cost) +
          '\t\tHighest Recorded Cost : {}\n'.format(highest_cost)
          )
    print('\tAccuracy:\n'
          '\t\tFinal Accuracy: {} %\n'.format(acc)
          )
    

# shuffle data
def shuffle_df(raw_data):
    return raw_data.sample(frac=1).reset_index(drop=True)


# split data and labels to (train, test[/ validation])
def get_shuffled_divided_data(features,labels):
    data_len = features.shape[0]

    TRAIN_SIZE = int(data_len * 0.6)

    train_df = features[:TRAIN_SIZE]
    train_labels = labels[:TRAIN_SIZE]

    test_df = features[TRAIN_SIZE:]
    test_labels = labels[TRAIN_SIZE:]

    return train_df, train_labels, test_df, test_labels


data_path = join("/home/cyberlab/Desktop/Shani's_ML/Deeplearning-Endo",'Dataset .xlsx')
dataset = pandas.read_excel(data_path)
dataset = dataset[wanted_columns]

dataset = shuffle_df(dataset)

dataset_labels = dataset['label']
dataset_features = dataset.drop(columns=['label'])

train_features, train_labels, test_features, test_labels = get_shuffled_divided_data(dataset_features, dataset_labels)

# initialize input and output vectors
X = tf.placeholder(tf.float32, [None, constants['features']])
Y = tf.placeholder(tf.float32, [None, constants['labels']])

# initialize weights and biases randomly
W = tf.Variable(0.001 * np.random.randn(constants['features'], constants['labels']).astype(np.float32))
b = tf.Variable(0.001 * np.random.randn(constants['labels']).astype(np.float32))

#TODO Y_ = tf.nn.softmax_cross_entropy_with_logits_v2(tf.add(tf.matmul(X, W), b))

Y_ = tf.nn.softmax(tf.add(tf.matmul(X, W), b))
#Loss and normalization
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(tf.clip_by_value(Y_, 1e-10, 1.0))))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=constants['alpha']).minimize(cost)

init = tf.global_variables_initializer()

# calculate bin size for training
constants['bins'] = int (train_features.shape[0]/constants['bin_size'])

training_start_time = dt.now()

# start tensor session
with tf.Session() as sess:
    train_labels = sess.run(tf.one_hot(train_labels, constants['labels']))
    test_labels = sess.run(tf.one_hot(test_labels, constants['labels']))

    sess.run(init)

    cost_hist = []

    # model training (optimizer)
    for epoch in range(constants['epochs']):
        for bi in range(constants['bins']):

            start_point = bi * epoch
            end_point = start_point + constants['bin_size']

            x = train_features[start_point: end_point]
            y = train_labels[start_point: end_point]

            sess.run(optimizer, feed_dict={X: x, Y: y})
            c = sess.run(cost, feed_dict={X: x, Y: y})

            if (epoch % 500 == 0 and epoch != 0) or (epoch == constants['epochs'] - 1):
                cost_hist.append(c)
                print('\rEpoch: {} Cost: {}'.format(str(epoch), str(c)))

            #numpy.save('FeaturesWeight.npy', W, allow_pickle=True, fix_imports=True)
    training_end_time = dt.now()

    # model testing
    correct_prediction = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = accuracy.eval({X: test_features, Y: test_labels}) * 100

    print('\rW: {}, b: {}\n'.format(W.eval(sess), b.eval(sess)))
    print_stats(acc, cost_hist, training_end_time, training_start_time)
   
    print('Confusion Matrix:')

    # confusion matrix
    conf_mat = tf.confusion_matrix(labels=tf.argmax(Y, 1), predictions=tf.argmax(Y_, 1), num_classes=2)
    conf_mat_to_print = sess.run(conf_mat, feed_dict={X: test_features, Y: test_labels})
    print(conf_mat_to_print)