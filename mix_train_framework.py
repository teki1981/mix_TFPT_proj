import tensorflow as tf
import torch as th
import numpy as np
import tfpyth

def print_grad(grad):
	print("torch print: ", grad)
	return grad

def backward_hook1(module, gin, gout):
	print("***1**** gin: ", gin)
	print("***1**** gout: ", gout)
	print("***1**** weight: {} grad: {}".format(module.weight, module.weight.grad))
def backward_hook2(module, gin, gout):
	print("***2**** gin: ", gin)
	print("***2**** gout: ", gout)
	print("***2**** weight: {} grad: {}".format(module.weight, module.weight.grad))

class linearZ(th.autograd.Function):
	@staticmethod
	def forward(ctx, input, weight):
		ctx.save_for_backward(input, weight)
		l2 = th.matmul(input, weight.t())
		return l2

	@staticmethod
	def backward(ctx, grad_output):
		input, weight = ctx.saved_tensors
		grad_input = grad_output.matmul(weight)
		grad_weight = grad_output.t().matmul(input)
		print("PT grad_weight: ", grad_weight)
		print("PT grad_input: ", grad_input)
		#weight.grad = grad_weight
		return grad_input,grad_weight

class PT_Model(th.nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()
		self.linear_1 = th.nn.Linear(input_size, input_size, bias=False)
		self.linear_2 = th.nn.Linear(input_size, output_size, bias=False)
		th.nn.init.constant_(self.linear_1.weight, 0.1)
		th.nn.init.constant_(self.linear_2.weight, 0.2)
		#self.linear_1.register_full_backward_hook(backward_hook1)
		#self.linear_2.register_full_backward_hook(backward_hook2)

	def forward(self, x):
		#logits = self.linear(x)
		y1 = self.linear_1(x)
		#print("PT linear_1 out :", y1)
		logits = self.linear_2(y1)
		#logits = linearZ.apply(x, self.linear.weight)
		return logits

class TF_Model(tf.Module):
	def __init__(self, input_size, output_size, name=None):
		super().__init__(name=name)
		#self.w = tf.Variable(tf.random.normal([input_size, output_size]), name='w')
		self.w_1 = tf.Variable(tf.ones([input_size, input_size]), name='w1')
		self.w_2 = tf.Variable(tf.ones([input_size, output_size]), name='w2')
	def __call__(self, x):
		y = tf.matmul(x, self.w_1)
		#tf.print("TF w_1 out : ", y)
		return tf.matmul(y, self.w_2)


#define tf input
tf_x = tf.Variable([[1., 2., 3.]]);
#define tf label
y_pred = tf.constant([[1.]])

#define tf model
tf_layer = TF_Model(input_size=3, output_size=2)

#define pt model
pt_layer = PT_Model(input_size=2, output_size=1)
def pytorch_expr(x):
	return pt_layer(x);
pt_func = tfpyth.eager_tensorflow_from_torch(pytorch_expr)

#define loss function
def loss_func(y, label):
	return tf.reduce_mean(tf.math.square(y - label), axis = -1);

tf_opt = tf.keras.optimizers.Adam()
pt_opt = th.optim.Adam(pt_layer.parameters(), lr=0.001)

def test_pytorch_in_tensorflow_eager_mode():
	
	with tf.GradientTape(persistent=True) as tape:
		tf_y = tf_layer(tf_x)
		print("tf_y: ", tf_y)
		pt_y = pt_func(tf_y);
		print("pt_y: ", pt_y)
		loss = loss_func(pt_y, y_pred);
		print("loss: ", loss)

	d_pt_y = tape.gradient(loss, pt_y)
	print("d_pt_y: ", d_pt_y)
	assert np.allclose(d_pt_y, [22])
	dx = tape.gradient(loss, tf_x)
	print("dx: ", dx)
	assert np.allclose(dx, [44, 44,44])
	return

def test_pytorch_in_tensorflow_graph_mode():
	
	@tf.function
	def train_step(x, pred):
		with tf.GradientTape(persistent=True) as tape:
			tf_y = tf_layer(x)
			#tf.print("tf_y: ", tf_y)
			#pt_y = tfpyth.tensorflow_from_torch(pytorch_expr, [tf_y], tf.float32)
			pt_y = tfpyth.tensorflow_from_torch(pt_layer, [tf_y], tf.float32)
			#tf.print("pt_y: ", pt_y)
			loss = loss_func(pt_y, pred);
			#tf.print("loss: ", loss)
		#d_pt_y = tape.gradient(loss, pt_y)
		#tf.print("d_pt_y: ", d_pt_y, " = 22")
		#dx = tape.gradient(loss, x)
		#tf.print("dx: ", dx)

		# tf model parameter update
		tf_params = tf_layer.trainable_variables;
		tf_grads = tape.gradient(loss, tf_params)
		#tf.print("TF grads : ", tf_grads)
		tf_opt.apply_gradients(zip(tf_grads, tf_params))
		#tf.print("TF weight after: ", tf_layer.trainable_variables)
		
		# pt model parameter update
		#pt_opt.step()
		return loss, pt_y

	#loss,y =  train_step(tf_x, y_pred)
	for step in range(20):
		loss,y =  train_step(tf_x, y_pred)
		pt_opt.step()
		#for name,parameter in pt_layer.named_parameters():
		#	print("PT parameter after, name:{}, weight:{}, grad:{}".format(name, parameter, parameter.grad))
		print("================================ step: ", step, " loss: ", loss.numpy(), " y: ", y.numpy())
	return

def test_pytorch_in_tensorflow_sess_mode():
	#tf.compat.v1.disable_eager_execution()
	tf_inp = np.random.rand(1,3) #[[1., 2., 3.]]
	#pt_inp = np.random.rand(1,2) #[[6., 6.]]
	tf_input = tf.compat.v1.placeholder(tf.float32, shape=(1,3), name="tf_x")
	#pt_input = tf.compat.v1.placeholder(tf.float32, shape=(1,2), name="pt_x")
	tf_y = tf_layer(tf_input)
	pt_y = tfpyth.tensorflow_from_torch(pytorch_expr, [tf_y], tf.float32)
	loss = loss_func(pt_y, y_pred)
	pt_y_grad = tf.gradients(loss, pt_y)
	x_grad = tf.gradients(y, tf_input)
	with tf.compat.v1.Session() as session:
		print("tf_input: ", session.run(tf_inp, {tf_input: tf_inp}))
		print("pt_input: ", session.run(pt_inp, {pt_input: pt_inp}))
		print("tf_y: ", session.run(tf_y, {tf_input: tf_inp}))
		print("pt_y: ", session.run(pt_y, {pt_input: pt_inp}))
		print("loss: ", session.run(loss, {}))
		print( session.run([y, pt_y_grad, x_grad], {tf_input: tf_inp, pt_input: pt_inp}), [12.0, [12.], [44.,44.,44.]] )
	return

if __name__ == "__main__":
	#test_pytorch_in_tensorflow_eager_mode()
	print("===============================")
	test_pytorch_in_tensorflow_graph_mode()
	#test_pytorch_in_tensorflow_sess_mode()
