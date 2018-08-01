import tensorflow as tf 

# initializer
# tf.constant_initializer(value=0, dtype=tf.float32)
# tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
# tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
# tf.random_uniform_initializer(minval=0, maxval=None, seed=None, dtype=tf.float32)
# tf.uniform_unit_scaling_initializer(factor=1.0, seed=None, dtype=tf.float32)
# tf.zeros_initializer(shape, dtype=tf.float32, partition_info=None)
# tf.ones_initializer(dtype=tf.float32, partition_info=None)
# tf.orthogonal_initializer(gain=1.0, dtype=tf.float32, seed=None)

# name scope中tf.Variable()的名字可以重复，它会自动修改，以免重复
# 而get variable的名字不能重复，并且不受namecope的名字影响
with tf.name_scope('a_name_scope'):
	initializer = tf.constant_initializer(value=[2, 1])
	var1 = tf.get_variable(name='var1', shape=[1, 2], dtype=tf.float32, initializer=initializer)
	var2 = tf.get_variable(name='var2', shape=[1, 2], dtype=tf.float32)
	var3 = tf.Variable(name='var3', initial_value=[1, 2], dtype=tf.float32)
	var4 = tf.Variable(name='var3', initial_value=[2, 1], dtype=tf.float32)

# variable scope 配合 get variable达到重复使用变量的效果
with tf.variable_scope('a_variable_scope', reuse=tf.AUTO_REUSE) as scope:
	# scope.reuse_variables()
	initializer = tf.constant_initializer(value=[3, 2])
	var5 = tf.get_variable(name='var5', shape=[1, 2], dtype=tf.float32, initializer=initializer)
	var5_reuse = tf.get_variable(name='var5')


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(var1))
	print(sess.run(var2))
	print(var3.name)
	print(sess.run(var3))
	print(var4.name)
	print(sess.run(var4))
	# [[2. 1.]]
	# [[-1.1295946  0.8657552]]
	# a_name_scope/var3:0
	# [1. 2.]
	# a_name_scope/var3_1:0
	# [2. 1.]
	print(sess.run(var5))
	print(sess.run(var5_reuse))
