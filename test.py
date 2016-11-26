import gym
import tensorflow as tf
from policygradient import PolicyGradient
env = gym.make('CartPole-v0')

print(env.observation_space.shape)
policy= PolicyGradient(None, [env.observation_space.shape[0], env.action_space.n], learningRate=1)

def rollout():
    observation = env.reset()
    payoff = 0
    for _ in range(1000):

        #env.render() # we can't render on CSX.
        action = env.action_space.sample()
        #action = 0
        action = policy.doAction(observation)
        observation, reward, done, info = env.step(action)
        #print(action, observation, reward, done, info)
        payoff += reward
        if done:
            #payoff -= 100 #penalize "losing"
            break
    return payoff

policy.rollout = rollout

for _ in range(100):
    policy.train(50, .5)

input("enter to close")
