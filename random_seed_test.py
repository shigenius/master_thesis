import tensorflow as tf
import random
r = random.randint(0, 1000000)

# tf.set_random_seed(r) # 乱数オペレータの生成後に乱数シードを設定
with tf.Graph().as_default():
    rand_op1 = tf.random_normal([3])
    rand_op2 = tf.random_normal([3])
    rand_op3 = tf.random_normal([3], seed=r)
    rand_op4 = tf.random_normal([3], seed=r)

    # tf.set_random_seed(0)  # 乱数オペレータの生成後に乱数シードを設定
    with tf.Session() as sess:
        print("op1-2:", sess.run([rand_op1, rand_op2]))
        print("op1-2:", sess.run([rand_op1, rand_op2]))
        print("op3-4:", sess.run([rand_op3, rand_op4]))
        print("op3-4:", sess.run([rand_op3, rand_op4]))
        print("op3-4:", sess.run([rand_op3, rand_op4]))
        print("op3-4:", sess.run([rand_op3, rand_op4]))