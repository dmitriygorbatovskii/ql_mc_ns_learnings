#!/usr/bin/env python3
from __future__ import print_function, division
import gym
from builtins import range
import random
import tensorflow as tf
import time
import pickle

#python3.8 -m tensorboard.main --logdir="/tmp/mylogs/eager1"
env = gym.make('gym_lgsvl:lgsvl-v0')



def ql():
    s = 0
    Q = []
    for i_episode in range(10000):
        observation = env.reset()
        episode_reward = 0
        j = 0
        while True:

            if observation in list(i[:6] for i in Q):
                if random.uniform(0, 1) > 0.1:
                    action = Q[j][6:].index(max(Q[j][6:]))
                else:
                    action = random.randint(0, 4)
            else:
                Q.append(observation + (list(random.random() for i in range(5))))
                action = Q[j][6:].index(max(Q[j][6:]))
            observation, reward, done, info = env.step(action)
            Q[j][action+6] += reward
            episode_reward += reward

            if done:
                s += episode_reward
                tf.summary.scalar("x_axis: steps, y_axis: reward", episode_reward, step=i_episode)
                writer.flush()
                break
    return Q



def mc():
    Q = []
    for i_episode in range(10000):
        observation = env.reset()
        episode_path = []
        actions = []
        episode_reward = 0
        j = 0
        while True:

            if i_episode == 0:
                episode_path.append(observation + (list(random.random() for i in range(5))))
                action = episode_path[j][6:].index(max(episode_path[j][6:]))
                observation, reward, done, info = env.step(action)

                episode_reward += reward
                actions.append(action)
                j += 1

            else:
                if observation in list(i[:6] for i in Q):
                    if random.uniform(0, 1) > 0.2:
                        for j in range(len(Q)):
                            if observation == Q[j][:6]:
                                action = Q[j][4:].index(max(Q[j][6:]))
                                episode_path.append(observation + [0,0,0,0,0])
                else:
                    action = random.randint(0, 4)
                    episode_path.append(observation + (list(random.random() for i in range(5))))
                actions.append(action)

                observation, reward, done, info = env.step(action)
                episode_reward += reward

            if done:
                tf.summary.scalar("x_axis: steps, y_axis: reward", episode_reward, step=i_episode)
                writer.flush()

                step_reward = episode_reward / len(actions)

                if i_episode == 0:
                    for i in range(len(episode_path)):
                        episode_path[i][actions[i] + 6] += step_reward
                    Q = episode_path

                else:
                    try:
                        for i in range(len(actions)-1):
                            if episode_path[i][:6] in list(j[:6] for j in Q):
                                for q in range(len(Q)):
                                    if episode_path[i][:6] == Q[q][:6]:
                                        Q[q][actions[i] + 6] += step_reward
                            else:
                                episode_path[i][actions[i] + 6] += step_reward
                                Q.append(episode_path[i])
                    except:
                        pass
                break
    return Q

def ns(steps = 5):
    Q = []
    s = 0
    for i_episode in range(10000):
        observation = env.reset()
        episode_step = 0
        episode_reward = []
        qwe = {}
        while True:
            if episode_step < steps:
                if observation in list(j[:6] for j in Q):
                    if random.uniform(0, 1) > 0.1:
                        for j in range(len(Q)):
                            if observation == Q[j][:6]:
                                action = Q[j][6:].index(max(Q[j][6:]))
                    else:
                        action = random.randint(0, 4)
                else:
                    action = random.randint(0, 4)
                observation, reward, done, info = env.step(action)
                s+=reward
                qwe[episode_step] = observation + [action, reward]
                episode_step += 1
                episode_reward.append(reward)
            else:
                a = qwe.pop(0)
                for i in range(1, len(qwe)+1):
                    qwe[i-1] = qwe[i]
                qwe.pop(len(qwe)-1)
                episode_step -= 1
                if a[:6] in list(j[:6] for j in Q):
                    for j in range(len(Q)):
                        if a[:6] == Q[j][:6]:
                            Q[j][action+6] += sum(episode_reward)
                else:
                    b = list(random.random() for i in range(5))
                    b[action] += sum(episode_reward)
                    episode_reward.pop(0)
                    Q.append(a[:6] + b)


            if done:
                episode_reward = episode_reward[len(episode_reward)-5:]
                for i in range(len(episode_reward)):
                    s+=episode_reward[i]
                tf.summary.scalar("x_axis: steps, y_axis: reward", s, step=i_episode)
                writer.flush()
                s = 0
                qwe.pop(0)
                episode_reward.pop(0)
                episode_reward = sum(episode_reward)
                for i in range(1, len(qwe)+1):
                    if qwe[i][:6] in list(j[:6] for j in Q):
                        for z in range(len(Q)):
                            if a[:6] == Q[z][:6]:
                                Q[z][qwe[i][6] + 6] += episode_reward
                    else:
                        b = list(random.random() for i in range(5))
                        b[qwe[i][6]] += episode_reward
                        Q.append(a[:6] + b)
                break
    return Q

writer = tf.summary.create_file_writer("/tmp/mylogs/ql")
with writer.as_default():
    data_ql = ql()
with open('data_ql.pickle', 'wb') as f:
    pickle.dump(data_ql, f)

writer = tf.summary.create_file_writer("/tmp/mylogs/mc")
with writer.as_default():
    data_mc = mc()
with open('data_mc.pickle', 'wb') as f:
    pickle.dump(data_mc, f)

writer = tf.summary.create_file_writer("/tmp/mylogs/ns")
with writer.as_default():
    data_ns = ns()
with open('data_ns.pickle', 'wb') as f:
    pickle.dump(data_ns, f)



