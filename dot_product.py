import numpy as np
import tensorflow as tf
import timeit        
from matplotlib import pyplot as plt
xaxis=[100, 1000, 4000, 8000, 10000]
steps=3

#numpy
times_np=[]
for dim in xaxis:
    print (dim)
    temp_times_np=np.zeros(steps)
    for j in range(steps):
        #A=np.random.normal(size=(dim, dim))
        #B=np.random.normal(size=(dim, dim))
        start_time = timeit.default_timer()
        out=np.dot(np.random.normal(size=(dim, dim)),np.random.normal(size=(dim, dim)))
        temp_times_np[j]=timeit.default_timer() - start_time
    temp_time_np=np.sum(temp_times_np)/temp_times_np.shape[0]
    times_np.append(temp_time_np)
del out

#all tensorflow
times_tf=[]
for dim in xaxis:
    print (dim)
    temp_times_tf=np.zeros(steps)
    for j in range(steps):
        A=tf.Variable(tf.random_normal([dim, dim], stddev=0.35),
                              name="weights")
        B=tf.Variable(tf.random_normal([dim, dim], stddev=0.35),
                              name="weights2")
        out=tf.matmul(A,B)
        start_time = timeit.default_timer()
        with tf.Session() as sess:
            # define your variables and tensors
            # ... initialization code ...

            sess.run(tf.global_variables_initializer())
            sess.run(out)
        temp_times_tf[j]=timeit.default_timer() - start_time
        del A, B, out
    temp_time_tf=np.sum(temp_times_tf)/temp_times_tf.shape[0]
    times_tf.append(temp_time_tf)

#mixed: using TensorFlow with external generated matrices
times_mix=[]
for dim in xaxis:
    print (dim)
    temp_times_mix=np.zeros(steps)
    for j in range(steps):
        A=np.random.normal(size=(dim, dim))
        B=np.random.normal(size=(dim, dim))
        a=tf.placeholder("double")
        b=tf.placeholder("double")

        out=tf.matmul(a,b)

        #with tf.Session() as sess:
        sess=tf.Session()
            # define your variables and tensors
            # ... initialization code ...
        sess.run(out, feed_dict={a: A, b: B})
        temp_times_mix[j]=timeit.default_timer() - start_time
        del A,B, out
    temp_time_mix=np.sum(temp_times_mix)/temp_times_mix.shape[0]

    times_mix.append(temp_time_mix)
plt.plot(xaxis_cr,times_np_cr,label='Numpy')

plt.plot(xaxis_cr,times_tf_cr,label='TensorFlow, matrices generated by TF')
plt.plot(xaxis_cr,times_mix_cr,label='TensorFlow, matrices passed on run')

plt.legend()
plt.ylabel('Execution time')
plt.xlabel('Matrix size')
plt.yscale('log')
plt.show()

