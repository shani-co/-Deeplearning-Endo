import tensorflow._api.v2.compat.v1 as tf, numpy as np, pandas
from datetime import datetime as dt
from os.path import join
tf.disable_v2_behavior()
constants = {
    'features': 58,
    'labels': 2,
    'alpha': 0.001,
    'epochs': 150000,
    'bin_size': 10000,
    'alpha': 0.0001,
    'epochs': 50000,
    'bin_size': 150000,
    'bins': None
}

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
    TRAIN_SIZE = int(data_len * 0.7)

    train_df = features[:TRAIN_SIZE]
    train_labels = labels[:TRAIN_SIZE]
    test_df = features[TRAIN_SIZE:]
    test_labels = labels[TRAIN_SIZE:]
    return train_df, train_labels, test_df, test_labels


data_path = join('C:\\Users\\שני כהן\\Desktop\\למידה עמוקה\\למידה עמוקה פרויקט','Dataset .xlsx')
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
hyp = tf.nn.softmax(tf.add(tf.matmul(X, W), b))
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(tf.clip_by_value(hyp, 1e-10, 1.0))))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=constants['alpha']).minimize(cost)
init = tf.global_variables_initializer()
# calculate bin size for training
constants['bins'] = int(train_features.shape[0] / constants['bin_size'])
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
        if (epoch % 500 is 0 and epoch is not 0) or (epoch is constants['epochs'] - 1):
            pass
            # cost_hist.append(c)
            # print('\rEpoch: {} Cost: {}'.format(str(epoch), str(c)))
            # print('\rW: {}, b: {}'.format(W.eval(sess), b.eval(sess)))
    training_end_time = dt.now()
    # model testing
    correct_prediction = tf.equal(tf.argmax(hyp, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = accuracy.eval({X: test_features, Y: test_labels}) * 100
    print_stats(acc, [1,2,3,4,5,6,7,8,9], training_end_time, training_start_time)
    # print_stats(acc, cost_hist, training_end_time, training_start_time)
    print('Confusion Matrix:')
    # confusion matrix
    conf_mat = tf.confusion_matrix(labels=tf.argmax(Y, 1), predictions=tf.argmax(hyp, 1), num_classes=2)
    conf_mat_to_print = sess.run(conf_mat, feed_dict={X: test_features, Y: test_labels})
    print(conf_mat_to_print)