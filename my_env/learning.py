import click
import random
from lgsvl_env import LgsvlEnv
import pickle


def Q_learning(alpha, gma, epoch, train):
    if train:
        Q = []
        env = LgsvlEnv()
        for i in range(epoch):
            state = env.reset()
            while True:
                for h in Q:
                    if h[2:] == [0, 0, 0, 0, 0]:
                        del h
                flag = True
                for row in Q:
                    if state == row[:3]:
                        current_step = [item for item in Q if item[:3] == state][0]
                        action = current_step[3:].index(max(current_step[3:]))
                        next_state, reward, done, info = env.step(action)
                        flag = False
                if flag == True:
                    row = state + [0, 0, 0, 0, 0]
                    Q.append(row)
                    current_step = [item for item in Q if item[:3] == state][0]
                    action = random.randint(0, 4)
                    next_state, reward, done, info = env.step(action)
                flag = True
                for row in Q:
                    if next_state == row[:3]:
                        next_step = [item for item in Q if item[:3] == next_state][0]
                        best_next_action = next_step[3:].index(max(next_step[3:]))
                        flag = False
                if flag == True:
                    row = next_state + [0, 0, 0, 0, 0]
                    Q.append(row)
                    next_step = [item for item in Q if item[:3] == next_state][0]
                    best_next_action = next_step[3:].index(max(next_step[3:]))
                for row in Q:
                    if state == row[:3]:
                        row[action + 3] = row[action + 3] + alpha * (
                                    reward + gma * (next_step[best_next_action + 3]) - row[action + 3])
                state = next_state
                if done:
                    break
        return Q
    else:
        with open('data.pickle', 'rb') as f:
            Q = pickle.load(f)
        env = LgsvlEnv()
        for i in range(epoch):
            state = env.reset()
            while True:
                current_step = [item for item in Q if item[:3] == state][0]
                action = current_step[3:].index(max(current_step[3:]))
                next_state, reward, done, info = env.step(action)
                if done:
                    break
        return False

def Monte_carlo():
    pass

def n_steps():
    pass


@click.command()
@click.option('--weather', default=0, help='set weather')
@click.option('--time', default=12, help='set time')
@click.option('--alpha', default=0.01, help='step')
@click.option('--gma', default=1, help='discount factor')
@click.option('--epoch', default=100, help='epochs')
@click.option('--agent', default='', help='Jaguar2015XE (Apollo 3.0),\n'
                                          'Lexus2016RXHybrid (Autoware),\n'
                                          'Lincoln2017MKZ (Apollo 5.0)')
@click.option('--train', default=True, help='True - train new model,'
                                            'False - demonstration of the trained model')
@click.option('--type', prompt='learning type', help='Q_learning, Monte_carlo, N_steps')
def main(weather, time, alpha, gma, epoch, agent, train, type):
    if type == 'Q_learning':
        return Q_learning(alpha, gma, epoch, train)
    elif type == 'Monte_carlo':
        return Monte_carlo()
    elif type == 'N_steps':
        return n_steps()


if __name__ == "__main__":
    data = main()
    if data:
        with open('data.pickle', 'wb') as f:
            pickle.dump(data, f)