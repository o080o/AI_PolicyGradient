import numpy as np
import tensorflow as tf

class PolicyGradient:

    # rollout - function that does a rollout, given the policy network
    # size - the shape of the network for policy network (including the input and output layer)
    def __init__(self, rollout, size):
        #self.sess = tf.Session()
        self.sess = tf.InteractiveSession()

        self.initWeightVal = 1
        self.rollout = rollout
        # create the nn for policy network
        lastLayer = -1
        x_in = 0
        for layer in size:
            y=0 # need to define outside of if-block
            if lastLayer >=0:
                w = tf.Variable(tf.random_uniform([lastLayer, layer], minval = -1*self.initWeightVal, maxval=self.initWeightVal), name="W")
                b = tf.Variable(tf.random_uniform([layer], minval = -1*self.initWeightVal, maxval=self.initWeightVal), name="B")
                y = tf.sigmoid(tf.matmul(x_in, w)+b)
            else:
                y = tf.placeholder(tf.float32, [None, layer], name="x")
                self.input = y
            lastLayer = layer
            x_in = y
        self.output = x_in
        self.probability = self.output/tf.reduce_sum(self.output) # may need to change this.

        learningRate = 0.1
        self.optimizer = tf.train.GradientDescentOptimizer(learningRate)

        init = tf.initialize_all_variables().run()

    def doAction(self,observation):
        #roll dice, then select based on our probability distribution
        probs = self.probability.eval({self.input: observation.reshape(1,len(observation))})
        probs = probs.flatten()
        roll = np.random.choice( len(probs), 1, p=probs)
        print(roll)
        return roll[0]
        

    def train(self, size, stepsize):
        # implements finite difference approach

        #collect all weights in a flat array
        parameters = tf.trainable_variables()
        print(parameters, parameters[0], parameters[1])

        referenceParameters = np.array([])
        for variable in parameters:
            referenceParameters = np.concatenate((referenceParameters, variable.eval().flatten()))
        print(referenceParameters)

        nparameters = len(referenceParameters)
        deltaReward = np.zeros(shape=(size, 1))
        deltaWeight = np.zeros(shape=(size, nparameters))

        reference = self.rollout()
        for i in range(size):
            delta = np.random.random(size=nparameters)
            deltaWeight[i] = referenceParameters + delta*stepsize
            deltaReward[i][0] = self.rollout()-reference #this is the *increase* in reward
            #variable.assign(value)
            #updateWeights(deltaWeights)

        #A = np.matmul(deltaWeight.transpose(),deltaWeight)
        #A = np.linalg.inv(A)
        #B = np.matmul(deltaWeight.transpose(),deltaReward)
        #gradient = np.matmul(A, B)

        # = (dWeight^T * dWeight)^-1 * (dWeight^T * dReward)
        gradient = np.matmul(np.linalg.inv(np.matmul(deltaWeight.transpose(),deltaWeight)),  np.matmul(deltaWeight.transpose(),deltaReward))
        gradient = gradient.reshape((nparameters))
        print("gradient shape", gradient.shape)

        #gradientsInput = np.zeros(shape=(nparameters))
        #gradientsInput = [0 for i in range(nparameters)]
        #print("gi",gradientsInput)
        gradientsInput = []

        # now shape the gradients into a per-variable tensor.
        start = 0

        
        apply_gradients = self.optimizer.apply_gradients(placeholder_gradients)
        for variable in parameters:
            print(variable)
            print(variable.get_shape())
            print( np.multiply.reduce(variable.get_shape() ))
            length = np.multiply.reduce(variable.get_shape())
            end = length + start
            print(length)
            print(gradient[i:length])
            gradientsInput.append( (gradient[start:end], variable) )
            start=end
        print(gradientsInput)

        # and apply them!!
        self.optimizer.apply_gradients(gradientsInput)

        # where each gradients[i] is a numpy array
        #for i, grad_var in enumerate(compute_gradients):

            #feed_dict[placeholder_gradients[i][0]] = gradients[i]
        #apply_gradients = optimizer.apply_gradients(placeholder_gradients)
        #apply_gradient.run(feed_dict=d)

        #gradients = placeholder(tf.float32, [None, dim])
        #optimizer.apply_gradients( zip(gradients, network_params))

