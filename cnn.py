# Lab 11 MNIST and Deep learning CNN
import cv2
import tensorflow as tf
import random,glob,re,os
import numpy as np
tf.set_random_seed(777)  # reproducibility


# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 30


class DataLoader:
    def __init__(self,path,extn='jpg'):
        self.flist = glob.glob(path+os.path.sep+'*.'+extn)
        self.fsize = len(self.flist)
        self.iter = 0

    def nex_batch(self,bsize):
        patrn = re.compile("(\d+)_(\d+)\.")
        image_list = []; label_list = []
        for i in range(self.iter,self.iter+bsize):
            if(i >= self.fsize):
                break
            rst = re.search(patrn,self.flist[i])
            if(rst is None):
                break
            img = cv2.imread(self.flist[i],cv2.IMREAD_UNCHANGED)
            img_class = [0,0,0,0]
            img_class[int(rst.groups()[0])] = 1

            image_list.append(img)
            label_list.append(img_class)

        self.iter += bsize
        return image_list, label_list

    def clear_iter(self):
        self.iter = 0

class CNNModel:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.keep_prob = tf.placeholder(tf.float32)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 128,128,3])
            self.Y = tf.placeholder(tf.float32, [None, 4])

            # L1 ImgIn shape=(?, 128, 128, 3)
            W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
            L1 = tf.nn.conv2d(self.X, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 4, 4, 1],
                                strides=[1, 4, 4, 1], padding='SAME')
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

            # L2 ImgIn shape=(?, 32, 32, 32)
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)

            # L3 ImgIn shape=(?, 16, 16, 64)
            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                                1, 2, 2, 1], padding='SAME')
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
            

            # L4 ImgIn shape=(?, 8, 8, 128)
            W4 = tf.Variable(tf.random_normal([3, 3, 128, 64], stddev=0.01))
            L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
            L4 = tf.nn.relu(L4)
            L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[
                                1, 2, 2, 1], padding='SAME')
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)

            #L5 ImgIn shape=(?, 4, 4, 64)
            L5 = tf.reshape(L4, [-1, 64 * 4 * 4])
            W5 = tf.get_variable("W5", shape=[64 * 4 * 4, 625],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([625]))
            L5 = tf.nn.relu(tf.matmul(L5, W5) + b5)
            L5 = tf.nn.dropout(L5, keep_prob=self.keep_prob)

            # L5 Final FC 625 inputs -> 10 outputs
            W6 = tf.get_variable("W6", shape=[625, 4],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b6 = tf.Variable(tf.random_normal([4]))
            self.logit = tf.matmul(L5, W6) + b6

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logit, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logit, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logit, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})

if __name__ == '__main__':
    dl = DataLoader('./train_set')
    print("total size",dl.fsize)
    with tf.Session() as sess:
        model = CNNModel(sess,'model')
        tf.global_variables_initializer().run()
        print('Learning Started!')

        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(dl.fsize/batch_size)
            for i in range(total_batch):
                batch_xs,batch_ys = dl.nex_batch(batch_size)
                c, _ = model.train(batch_xs,batch_ys)
                avg_cost += c / total_batch
            dl.clear_iter()
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
        print('Learning Finished!')
        
        # Test model and check accuracy

        tl = DataLoader('./test_set')
        test_img,test_lbl = tl.nex_batch(40)
        print('Accuracy:', model.get_accuracy(test_img,test_lbl))

        result_lbl = np.argmax(model.predict(test_img),axis=1)
        result_path = './test_set/classified'
        if(not os.path.exists(result_path)):
            os.makedirs(result_path)

        save_name = '{0}/{1}_{2}.jpg'
        for idx,label in enumerate(result_lbl):
            cv2.imwrite(save_name.format(result_path,label,idx),test_img[idx])
