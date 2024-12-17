import tensorflow as tf
import torch as th
import numpy as np
import tfpyth


def test_pytorch_in_tensorflow_eager_mode():
	tf.compat.v1.enable_eager_execution()
	#tfe = tf.contrib.eager

	tf_a = tf.Variable(1.0);
	tf_b = tf.Variable(3.0);

	def pytorch_expr(a, b):
		th_a = th.tensor(3.0)
		th_b = th.tensor(4.0)
		return th_a * a + th_b * b * b
	pt_func = tfpyth.eager_tensorflow_from_torch(pytorch_expr)

	test_y = tfpyth.eager_tensorflow_from_torch(pytorch_expr)
	assert tf.math.equal(test_y(tf_a, tf_b), 39.0)

	with tf.GradientTape(persistent=True) as tape:
		y = pt_func(tf_a, tf_b);
		print("y: ", pt_func(tf_a, tf_b))
	dx = tape.gradient(y, [tf_a, tf_b])
	print("dx: ", dx)
	
	assert np.allclose(dx, [3.0, 24.0])
	
	tf.compat.v1.disable_eager_execution()
	return
	
def test_pytorch_in_tensorflow_graph_mode():
	tf.compat.v1.disable_eager_execution()
	session = tf.compat.v1.Session();
	
	def pytorch_expr(a, b):
		th_a = th.tensor(3.0)
		th_b = th.tensor(4.0)
		return th_a * a + th_b * b * b


	tf_a = tf.compat.v1.placeholder(tf.float32, name="a")
	tf_b = tf.compat.v1.placeholder(tf.float32, name="b")
	y = tfpyth.tensorflow_from_torch(pytorch_expr, [tf_a, tf_b], tf.float32)
	x_grad = tf.gradients([y], [tf_a, tf_b], unconnected_gradients="zero")

	assert np.allclose(session.run([y, x_grad[0], x_grad[1]], {tf_a: 1.0, tf_b: 3.0}), [39.0, 3.0, 24.0])
	print(session.run([y, x_grad], {tf_a: 1.0, tf_b: 3.0}), [39.0, [3.0, 24.0]])

if __name__ == "__main__":
	test_pytorch_in_tensorflow_eager_mode()
	test_pytorch_in_tensorflow_graph_mode()
