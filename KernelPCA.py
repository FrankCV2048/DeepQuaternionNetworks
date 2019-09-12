import tensorflow as tf
def pca_data(input):


    m, n = 10, 1000
    # assert not tf.assert_less(128, n)
    mean = tf.reduce_mean(input, axis=1)
    x_new = input - tf.reshape(mean, (-1, 1))
    # 协方差矩阵
    cov = tf.matmul(x_new, x_new, transpose_a=True) / m
    e, v = tf.linalg.eigh(cov, name="eigh")

    e_index_sort = tf.nn.top_k(e, sorted=True, k=128)[1]
    v_new = tf.gather(v, indices=e_index_sort)
    pca = tf.matmul(x_new, v_new, transpose_b=True,name='PCATra')#这一层没问题
    return pca

def hash(input):
    input1=input[:,1:input.shape[1]]#1
    input2=tf.reshape(input[:,0],[-1,1],name='input_hash_1')
    hash_concat=tf.concat([input1, input2], axis=1,name='hash_concat')
    input=tf.subtract(input,hash_concat,name='hash_sub')

    input=tf.add(input,tf.abs(input),name='hash_add')

    output=tf.sign(input,name='hash_final')
    return output





