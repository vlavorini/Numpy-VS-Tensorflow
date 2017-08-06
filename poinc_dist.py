def dist_matrix_np(points):
    u'''
    if expanded, the following is equal to:
    
    expd=np.expand_dims(points,2)
    tiled=np.tile(expd, points.shape[0])
    trans=np.transpose(points)
    num=np.sum(np.square(trans-tiled), axis=1)
    #num
    den1=1-np.sum(np.square(points),1)
    dend=np.expand_dims(den1,1)
    den1M=np.matrix(dend)
    den=den1M * den1M.T
    
    return 1+2*np.divide(num, den)
    '''
    return 1+2*np.divide(np.sum(np.square(np.transpose(points)-np.tile(np.expand_dims(points,2), points.shape[0])), axis=1), np.matrix(np.expand_dims(1-np.sum(np.square(points),1),1)) * np.matrix(np.expand_dims(1-np.sum(np.square(points),1),1)).T)

def dist_matrix_tf(points):
    
    u'''
    if expanded, is equal to:
    ptf = tf.placeholder("double")

    expd=tf.expand_dims(ptf,2) # from (n_emb x emb_dim) to (n_emb x emb_dim x 1)
    tiled=tf.tile(expd, [1,1,tf.shape(ptf)[0]]) # copying the same matrix n times
    trans=tf.transpose(ptf)
    num=tf.reduce_sum(tf.squared_difference(trans,tiled), 1)
    den1=1-tf.reduce_sum(tf.square(ptf),1)
    den1=tf.expand_dims(den1, 1)
    den=tf.matmul(den1, tf.transpose(den1))

    tot=1+2*tf.div(num, den)
    '''
    ptf = tf.placeholder("double")
    tot=1+2*tf.div(tf.reduce_sum(tf.squared_difference(tf.transpose(ptf), tf.tile(tf.expand_dims(ptf,2), [1,1,tf.shape(ptf)[0]])), 1), tf.matmul(tf.expand_dims(1-tf.reduce_sum(tf.square(ptf),1), 1), tf.transpose(tf.expand_dims(1-tf.reduce_sum(tf.square(ptf),1), 1))))
    sess = tf.Session()
    return sess.run(tot, feed_dict={ptf: points})


def dist_matrix_tf_all(nvecs):
    with tf.device('/gpu:0'):
        ptf=tf.Variable(tf.random_uniform([nvecs, 2], minval=-0.99, maxval=0.99),
                                  name="weights")
        tot=1+2*tf.div(tf.reduce_sum(tf.squared_difference(tf.transpose(ptf), tf.tile(tf.expand_dims(ptf,2), [1,1,tf.shape(ptf)[0]])), 1), tf.matmul(tf.expand_dims(1-tf.reduce_sum(tf.square(ptf),1), 1), tf.transpose(tf.expand_dims(1-tf.reduce_sum(tf.square(ptf),1), 1))))
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    return (sess.run(tot))



data=np.random.rand(1182,2)

import timeit
#("Y0 = myfunctions.func1(X0)", setup="import myfunctions; X0 = 1")
times_np=[]
times_tf=[]
times_tf_all=[]
xaxis=[50, 100, 500, 1000, 5000, 10000, 15000, 20000]
steps=3
for i in xaxis:
    print (i)
    temp_times_np=np.zeros(steps)
    temp_times_tf=np.zeros(steps)
    temp_times_tf_all=np.zeros(steps)

    for j in range(steps):
        #numpy
        start_time = timeit.default_timer()
        dist_matrix(data[:i])
        temp_times_np[j]=timeit.default_timer() - start_time
        
        #tensorflow
        start_time = timeit.default_timer()
        dist_matrix_tf(data[:i])
        temp_times_tf[j]=timeit.default_timer() - start_time

        #tensorflow, self generation
        start_time = timeit.default_timer()
        dist_matrix_tf_all(i)
        temp_times_tf_all[j]=timeit.default_timer() - start_time

    temp_time_np=np.sum(temp_times_np)/temp_times_np.shape[0]
    temp_time_tf=np.sum(temp_times_tf)/temp_times_tf.shape[0]
    temp_time_tf_all=np.sum(temp_times_tf_all)/temp_times_tf_all.shape[0]

    times_np.append( temp_time_np)
    times_tf.append( temp_time_tf)
    times_tf_all.append( temp_time_tf_all)
from matplotlib import pyplot as plt
plt.plot(xaxis,times_np,label='Numpy')
plt.plot(xaxis,times_tf,label='TensorFlow, vectors passed on GPU')
plt.plot(xaxis,times_tf_all,label='TensorFlow, vectors generated in GPU')
plt.legend()
plt.ylabel('Execution time')
plt.xlabel('number of vectors')
#plt.yscale('log')
plt.show()
