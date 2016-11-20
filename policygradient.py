class PolicyGradient:

    # rollout - function that does a rollout, given the policy network
    # size - the shape of the network for policy network
    # actions - list of possible actions
    def __init__(self, rollout, size, actions):
        self.rollout = rollout
        # create the nn for policy network
        # ...

    def train(self):
        payoff = self.rollout()
        gradients = [payoff...] # how to do?
        optimizer.apply_gradient(gradients)
