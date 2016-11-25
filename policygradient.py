import numpy as np

class PolicyGradient:

    # rollout - function that does a rollout, given the policy network
    # size - the shape of the network for policy network (including the input and output layer)
    def __init__(self, rollout, size):
        self.rollout = rollout
        # create the nn for policy network
        x = tf.placeholder(tf.float32, [None, size[0]], name="x")
        lastLayer = -1
        x_in = 0
        for layer in size:
            w = tf.Variable(tf.random.uniform([lastLayer, layer], minval = -1*self.initWeightVal, maxval=self.initWeightVal))
            b = tf.Variable(tf.random_uniform
            y = tf.sigmoid(tf.matmul(x_in, w)+b)
            lastLayer = layer
            x_in = y
        self.output = x_in
        self.probability = output/tf.reduce_sum(self.output)

        learningRate = 0.1
        self.optimizer = tf.train.GradientDescentOptimizer(learningRate)

    def doAction(observation):
        #roll dice, then select based on our probability distribution
        

    def train(self, size, stepsize):
        # implements finite difference approach
        deltaReward = np.zeros(shape=(nparameters, height))
        deltaWeight = np.array()
        reference = self.rollout()
        for i in range(size):
            deltaWeight[i] = ...
            deltaReward[i] = self.rollout()-reference #this is the *increase* in reward
            updateWeights(deltaWeights)
        gradient = (deltaWeights*deltaWeights.transpose())^-1 * (deltaWeight*deltaReward.transpose())

        optimizer.apply_gradient(gradient)

        # where each gradients[i] is a numpy array
        for i, grad_var in enumerate(compute_gradients):
            feed_dict[placeholder_gradients[i][0]] = gradients[i]
        apply_gradients = optimizer.apply_gradients(placeholder_gradients)
        apply_gradient.run(feed_dict=d)


        gradients = placeholder(tf.float32, [None, dim])
        optimizer.apply_gradients( zip(gradients, network_params))

