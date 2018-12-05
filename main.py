import tensorflow as tf
import  numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
PATH = os.path.dirname(os.path.abspath(__file__))
DIR=os.path.join(PATH,"data")
MDIR=os.path.join(DIR,"modelsave")
from model import Classifier
e = len([file for file in os.listdir(MDIR) if os.path.isdir(os.path.join(MDIR,file)) ])

def main():
    iteration = 3001
    statistic_amount=200
    batch=128
    premodel=True
    mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)


    print('输入数据:', mnist.train.images)
    print('输入数据打印shape:', mnist.train.images.shape)
    import pylab
    im = mnist.train.images[1]
    im = im.reshape(-1, 28)
    #pylab.imshow(im)
    #pylab.show()
    print('输入数据打印shape:', mnist.test.images.shape)
    print('输入数据打印shape:', mnist.validation.images.shape)

    with tf.device("/gpu:0"):
        config=tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True


        with tf.Session(config=config) as sess:

            #global_episode=tf.Variable(0,dtype=tf.int32,trainable=False)

            classifier=Classifier(sess,10)
            tf.summary.scalar('acc', classifier.accuracy)
            tf.summary.scalar('loss', classifier.loss)
            summaryit = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(logdir=DIR, graph=sess.graph)


            saver = tf.train.Saver()
            time = len([file for file in os.listdir(MDIR) if os.path.isdir(os.path.join(MDIR, file))]) -1
            sess.run(tf.global_variables_initializer())
            if time<0:
                print("no model")
                premodel=False
            elif classifier.load(saver, MDIR):
                print("have model")
            else:
                print("still have some problems")


            for i in range(0,iteration):
                X,Y=mnist.train.next_batch(batch)
                classifier.caculate(X,Y)
                if i % statistic_amount==0:
                    train =classifier.statistic(mnist.validation.images,mnist.validation.labels,i,summaryit)
                    summary_writer.add_summary(train,i)



            classifier.test(mnist.test.images,mnist.test.labels)
            summary_writer.close()
            nDir=os.path.join(MDIR,str(e))
            if not os.path.exists(nDir):
                os.mkdir(nDir)
            classifier.save(saver,nDir)


























if __name__=="__main__":
    main()