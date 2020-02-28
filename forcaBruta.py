!pip install cmake 'gym[atari]' scipy

import gym

env = gym.make("Taxi-v3").env

env.s = 328  #Informando o estado do ambiente

epochs = 0
penalties, reward = 0, 0

frames = [] #Animação

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1
    
    #Pega cada frame renderizado para animação
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1
    
    
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))

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