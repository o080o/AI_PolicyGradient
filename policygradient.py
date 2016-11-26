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
                y = tf.matmul(x_in, w)
                # uncomment to use bias AND weights. (doesn't seem necessary)
                #b = tf.Variable(tf.random_uniform([layer], minval = -1*self.initWeightVal, maxval=self.initWeightVal), name="B")
                #y = tf.sigmoid(tf.matmul(x_in, w)+b)
            else: # first layer, aka the input
                y = tf.placeholder(tf.float32, [None, layer], name="x")
                self.input = y
            lastLayer = layer
            x_in = y
        #self.output = tf.sigmoid(x_in) # clamp to 0-1 range
        #self.probability = self.output/tf.reduce_sum(self.output) # may need to change this.
        self.output = x_in

        self.optimizer = tf.train.GradientDescentOptimizer(learningRate)
        init = tf.initialize_all_variables().run()

    # given an observation return an action (this is *stochastic*, sampling
    # from a distribution. this introduces a *lot* of variance on the reward
    # for any given parameterization of the policy (the weights) )
    def doStochasticAction(self, observation):
        #roll dice, then select based on our probability distribution
        probs = self.probability.eval({self.input: observation.reshape(1,len(observation))})
        probs = probs.flatten()
        roll = np.random.choice( len(probs), 1, p=probs)
        return roll[0]

    # given an observation, returns an action (this is *deterministic*, aka not
    # sampling from a distribution)
    def doAction(self,observation):
        output = self.output.eval({self.input: observation.reshape(1,len(observation))})
        output = output.flatten()
        if output[0] > 0:
            return 1
        else:
            return 0

    # reshapes a flat array back to the apropriate shapes for each parameter.
    #   for instance, if the first parameter is 3x4, then the first 12 elements
    #   in the flat array are reshaped into a 3x4 matrix, and so on.
    def reshapify(self,flatArray, parameters):
        start = 0
        reshaped = []
        for variable in parameters:
            length = np.multiply.reduce(variable.get_shape()) # the number of elements in this parameter
            end = length + start
            reshaped.append( np.array(flatArray[start:end], dtype=np.float32).reshape(variable.get_shape()) )
            start=end
        return reshaped



    # updates the actual value of the weights in the network, given a flat list
    # of parameters.
    def updateWeights(self, newWeight, parameters):
        weights = self.reshapify(newWeight, parameters) #reshape the flat list
        for i in range(len(parameters)): # for every trainable variable,
            self.sess.run(parameters[i].assign( weights[i] )) # assign its value.

    # implement policy updates using the finite difference approach
    def finiteDifference(self, size, stepsize):

        #get all the trainable variables
        parameters = tf.trainable_variables()
        
        #collect all weights in a flat array
        referenceParameters = np.array([])
        for variable in parameters: # for every variable, we get their parameters and flatten them, then concatenate to the flat array
            referenceParameters = np.concatenate((referenceParameters, variable.eval().flatten()))

        nparameters = len(referenceParameters)
        deltaReward = np.zeros(shape=(size, 1))             #this is the new rewards
        deltaWeight = np.zeros(shape=(size, nparameters))   #this is the new weights

        reference = self.rollout(render=True)               # this is the baseline performance

        # do several rollouts in order to calculate the gradient
        total = 0 #used for logging
        for i in range(size):
            delta = np.random.random(size=nparameters)*2*stepsize - stepsize #calculate a random offset
            newWeight = referenceParameters + delta         # calculate a new parameterization around the baseline
            deltaWeight[i] = delta                          # save the delta for gradient calculation
            self.updateWeights(deltaWeight[i], parameters)  # set the parameters in the NN
            payoff = self.rollout()                         # and do a rollout/trajectory/episode
            deltaReward[i][0] = payoff-reference            # calculate the increase in reward
            total += payoff         

        print(total/size, reference) #prints the average performance and the baseline for debugging

        self.updateWeights(referenceParameters, parameters) #return to base model for gradient update

        # taken from paper: g = (dWeight^T * dWeight)^-1 (dWeight^T * dReward)
        gradient = np.matmul(np.linalg.inv(np.matmul(deltaWeight.transpose(),deltaWeight)),  np.matmul(deltaWeight.transpose(),deltaReward))
        gradient = gradient.reshape((nparameters))  # reshape to flat list

        shapedGradients = self.reshapify(gradient, parameters)  #and reshape again!
        gradientsInput = zip(shapedGradients, parameters)       # and finally zip with the variable list
        self.optimizer.apply_gradients(gradientsInput)          # and finally apply them!

    # rather than following gradients, this implements a greedy random walk of
    # the parameter space, looking for better policy parameters (aka better
    # weights)
    # used as a baseline comparison
    def greedySearch(self, size, stepsize):

        # get all the trainable variables
        parameters = tf.trainable_variables()

        #collect all weights in a flat array
        referenceParameters = np.array([])
        for variable in parameters: # for every variable, we get their parameters and flatten them, then concatenate to the flat array
            referenceParameters = np.concatenate((referenceParameters, variable.eval().flatten()))

        nparameters = len(referenceParameters)
        deltaReward = np.zeros(shape=(size, 1))             #this is the new rewards
        newWeights = np.zeros(shape=(size, nparameters))   #this is the new weights

        reference = self.rollout(render=True)               # this is the baseline performance

        # do several rollouts in order to calculate the gradient
        total = 0 #used for logging
        print("reference:", reference)
        for i in range(size):
            delta = np.random.random(size=nparameters)*2*stepsize - stepsize    # calculate offset
            newWeights[i] = referenceParameters + delta         # calculate a new parameterization around the baseline
            self.updateWeights(newWeights[i], parameters)  # set the parameters in the NN
            payoff = self.rollout()                         # and do a rollout/trajectory/episode
            deltaReward[i][0] = payoff-reference            # calculate the increase in reward
            total += payoff
            print("episode", i, "reward", payoff)


        #select the best run to use as the new policy parameters
        maxPayoff = 0 #aka, no improvement. since all our rewards are stored as offsets, anything greater than 0 is an improvement
        bestIteration = -1 # start with no best Iteration.
        for i in range(size):                   # for every rollout,
            if deltaReward[i][0] > maxPayoff:   #if it's an improvement over the best so far, select it
                bestIteration = i
                maxPayoff = deltaReward[i][0]

        if bestIteration >= 0: #aka, there was at least *one* better rollout than reference
            self.updateWeights(newWeights[bestIteration], parameters) #update to better model
            print("best=", bestIteration)
        else:
            self.updateWeights(referenceParameters, parameters) #restore base model
            print("no improvement")
