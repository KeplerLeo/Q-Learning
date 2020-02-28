!pip install cmake 'gym[atari]' scipy

import gym

env = gym.make("Taxi-v3").env

env.s = 328  #Informando o estado do ambiente

import numpy as np
q_table = np.zeros([env.observation_space.n, env.action_space.n])

%%time

"""Treino do agente"""
import random
from IPython.display import clear_output

# Hyperparametros
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# Métricas
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() #Explorar espaços de ações
        else:
            action = np.argmax(q_table[state]) #Explorar valores aprendidos

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episódio: {i}")

print("Treino finalizado.\n")

q_table[328]

"""Avaliação do agente após o treinamento com Q-Learning"""

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    frames = [] #Animação
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1
     
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
            }
        )
        epochs += 1

    total_penalties += penalties
    total_epochs += epochs
    
from IPython.display import clear_output
from time import sleep

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Passos: {i + 1}")
        print(f"Estado: {frame['state']}")
        print(f"Ação: {frame['action']}")
        print(f"Recompensa: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)
print(f"Resultado após {episodes} episódios:")
print(f"Média de tempo por episódio: {total_epochs / episodes}")
print(f"Média de penalidades por episódio: {total_penalties / episodes}")