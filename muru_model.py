import os
import numpy as np
import timeit
import theano
import theano.tensor as T
import pickle

def load_data(dataset):
	file  = open(dataset,'rb')
	data = pickle.load(file)
	def shared_dataset(data_xy, borrow=True):
		data_x , data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX ),
                                 borrow=borrow)
		shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
		return shared_x, T.cast(shared_y,'int32')
	train_set_x, train_set_y = shared_dataset(data)
	val = [(train_set_x, train_set_y)]
	return val
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)


        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class hiddenLayer(object):
	def __init__(self,rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
		self.input = input
		if W is None:
			W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtpye = theano.config.floatX)
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4
			W = theano.shared(value = W_values, name = 'W', borrow= True)
		if b is None:
			b_values = np.zeros((n_out,),dtype = theano.config.floatX)
			b = theano.shared(value=b_values, name = 'b', borrow = True)
		self.W = W
		self.b = b
		lin_output = T.dot(input, self.W) + self.b
		self.output = (lin_output if activation is None else activation(lin_output))
		self.params = [self.W, self.b]

class model(object):
	def __init__(self,rng, input , n_in, hidden, n_out):
		self.hiddenlayer = hiddenLayer(rng =rng, input = input, n_in = n_in, n_out=n_hidden, activation=T.tanh)
		self.logregression = LogisticRegression(input = self.hiddenlayer.output, n_in = n_hidden, n_out = n_out)
		self.L1 = abs(self.hiddenlayer.W.sum() + abs(self.logregression.W).sum())
		self.L2_sqr = ((self.hiddenlayer.W**2).sum() + (self.logregression.W**2).sum())
		self.negative_log_likelihood = (self.logregression.negative_log_likelihood)
		self.errors = self.logregression.errors
		self.params = self.hiddenLayer.params + self.logregression.params
		self.input = input


def test_model(learning_rate = 0.01, L1_reg = 0.00, L2_reg = 0.0001, n_epochs = 1000, dataset = 'dataset.dat', batch_size=20, n_hidden=500):
	datasets = load_data(dataset)
	train_set_x, train_set_y = datasets

	n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
	index = T.lscalar()
	x = T.matrix('x')
	y = T.ivector('y')

	rng = np.random.RandomState(1234)
	classifier = model(rng  = rng ,input=X, n_in = 20*43, n_hidden = n_hidden, n_out = 5 )
	cost = (classifier.negative_log_likelihood(y) + L1_reg*classifier.L1 + L2_reg*classifier.L2_sqr )
	gparams = [T.grad(cost,param) for param in classifier.params]
	updates = [(param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)]
	train_model = theano.function(inputs = [index], outputs = cost, updates = updates, 
									givens = {
											x : train_set_x[index * batch_size : (index + 1 ) * batch_size],
											y : train_set_y[index * batch_size : (index + 1) * batch_size]
									})
	print 'training'

	'''patience = 10000
	patience_increase = 2
	improvement_threshold = 0.995
	validation_frequency = min(n_train_batches, patience // 2)
	best_validation_loss = numpy'''
	epoch = 0
	done_looping = False
	while (epoch < n_epochs) and (not done_looping):
		epoch  = epoch +1
		for minibatch_index in range(n_train_batches):
			minibatch_avg_cost = train_model(minibatch_index)
		if epoch % 100 == 0:
			print 'epoch ',epoch,minibatch_index
				
if __name__ == '__main__':
	test_model()
	print 'done'