import gym
import tensorflow as tf
env = gym.make('CartPole-v0')
env.reset()

def rollout():
    payoff = 0
    for _ in range(1000):

        #env.render() # we can't render on CSX.
        action = env.action_space.sample()
        action = 0
        observation, reward, done, info = env.step(action)
        print(action, observation, reward, done, info)
        payoff += reward
        if done:
            payoff -= 100 #penalize "losing"
            break
    print("rollout done.",payoff)

PG = PolicyGradient(rollout, shape, env.action_space)
for _ in range(100):
    PG.train()
