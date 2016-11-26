import numpy as np
import tensorflow as tf

class PolicyGradient:

    # rollout - function that does a rollout, given the policy network
    # size - the shape of the network for policy network (including the input and output layer)
    def __init__(self, rollout, size, learningRate=1):
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

        self.optimizer = tf.train.GradientDescentOptimizer(learningRate)

        init = tf.initialize_all_variables().run()

    def doAction(self,observation):
        #roll dice, then select based on our probability distribution
        probs = self.probability.eval({self.input: observation.reshape(1,len(observation))})
        probs = probs.flatten()
        roll = np.random.choice( len(probs), 1, p=probs)
        return roll[0]
        

    def reshapify(self,flatArray, parameters):
        start = 0
        reshaped = []
        for variable in parameters:
            length = np.multiply.reduce(variable.get_shape()) # the number of elements in this parameter
            end = length + start
            reshaped.append( np.array(flatArray[start:end], dtype=np.float32).reshape(variable.get_shape()) )
            start=end
        return reshaped



    def updateWeights(self, newWeight, parameters):
        weights = self.reshapify(newWeight, parameters)
        for i in range(len(parameters)):
            self.sess.run(parameters[i].assign( weights[i] ))

    def finiteDifference(self, size, stepsize):
        # implements finite difference approach

        #collect all weights in a flat array
        parameters = tf.trainable_variables()

        referenceParameters = np.array([])
        for variable in parameters:
            referenceParameters = np.concatenate((referenceParameters, variable.eval().flatten()))

        nparameters = len(referenceParameters)
        deltaReward = np.zeros(shape=(size, 1))
        deltaWeight = np.zeros(shape=(size, nparameters))

        reference = self.rollout()
        total = 0
        for i in range(size):
            delta = np.random.random(size=nparameters)
            deltaWeight[i] = referenceParameters + delta*stepsize
            payoff = self.rollout()
            total += payoff
            deltaReward[i][0] = payoff-reference #this is the *increase* in reward
            #variable.assign(value)
            self.updateWeights(deltaWeight[i], parameters)
        print(total, reference)



        self.updateWeights(referenceParameters, parameters) #return to base model for gradient update

        gradient = np.matmul(np.linalg.inv(np.matmul(deltaWeight.transpose(),deltaWeight)),  np.matmul(deltaWeight.transpose(),deltaReward))
        gradient = gradient.reshape((nparameters))

        shapedGradients = self.reshapify(gradient, parameters)
        gradientsInput = zip(shapedGradients, parameters)
        # and apply them!!
        self.optimizer.apply_gradients(gradientsInput)

    def train(self, size, stepsize):
        # implements finite difference approach

        #collect all weights in a flat array
        parameters = tf.trainable_variables()

        referenceParameters = np.array([])
        for variable in parameters:
            referenceParameters = np.concatenate((referenceParameters, variable.eval().flatten()))
        print(referenceParameters)

        nparameters = len(referenceParameters)
        deltaReward = np.zeros(shape=(size, 1))
        deltaWeight = np.zeros(shape=(size, nparameters))

        reference = self.rollout()
        total = 0
        for i in range(size):
            delta = np.random.random(size=nparameters)
            deltaWeight[i] = referenceParameters + delta*stepsize
            payoff = self.rollout()
            total += payoff
            deltaReward[i][0] = payoff-reference #this is the *increase* in reward
            #variable.assign(value)
            self.updateWeights(deltaWeight[i], parameters)
        print(total)


        #select the best run, and go from there.
        maxPayoff = 0
        bestIteration = -1
        for i in range(size):
            if deltaReward[i][0] > maxPayoff:
                bestIteration = i
                maxPayoff = deltaReward[i][0]

        self.updateWeights(deltaWeight[bestIteration], parameters) #return to base model for gradient update
