import tensorflow as tf 

var = tf.Variable(0) # our first variable in the 'global_variable' set

add_operation = tf.add(var, 1)
update_operation = tf.assign(var, add_operation)

with tf.Session() as sess:
	# once define variable, you have to initialzie them by doing this
	sess.run(tf.global_variables_initializer())
	for _ in range(3):
		sess.run(update_operation)
		print(sess.run(var))