class PolicyGradient:

    # rollout - function that does a rollout, given the policy network
    # inputs - the input nodes
    # size - the shape of the network for policy network
    def __init__(self, rollout, inputs, size):
        self.rollout = rollout
        # create the nn for policy network
        # ...

    def train(self):
        payoff = self.rollout()
        gradients = [payoff]
        optimizer.apply_gradient(gradients)
