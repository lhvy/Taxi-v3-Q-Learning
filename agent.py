import gym
import numpy
import random
from os import system, name
from time import sleep

# Define function to clear console window.
def clear(): 
  
    # Clear on Windows.
    if name == 'nt': 
        _ = system('cls')
  
    # Clear on mac and linux. (os.name is 'posix') 
    else: 
        _ = system('clear')

clear()

"""Setup"""

env = gym.make("Taxi-v3").env # Setup the Gym Environment

# Make a new array filled with zeros.
# The array will be 500x6 as there are 500 states and 6 actions.
q_table = numpy.zeros([env.observation_space.n, env.action_space.n])

training_episodes = 20000 #Amount of times to run environment while training.
display_episodes = 10 # Amount of times to run environment after training.

# Hyperparameters
alpha = 0.1 # Learning Rate
gamma = 0.6 # Discount Rate
epsilon = 0.1 # Chance of selecting a random action instead of maximising reward.

# For plotting metrics
all_epochs = []
all_penalties = []

"""Training the Agent"""

for i in range(training_episodes):
    state = env.reset()
    done = False
    penalties, reward, = 0, 0
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Pick a new action for this state.
        else:
            action = numpy.argmax(q_table[state]) # Pick the action which has previously given the highest reward.

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action] # Retreive old value from the q-table.
        next_max = numpy.max(q_table[next_state])

        # Update q-value for current state.
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10: # Checks if agent attempted to do an illegal action.
            penalties += 1

        state = next_state
        
    if i % 100 == 0: # Output number of completed episodes every 100 episodes.
        print(f"Episode: {i}")

print("Training finished.\n")

"""Display and evaluate agent's performance after Q-learning."""

total_epochs, total_penalties = 0, 0

for _ in range(display_episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = numpy.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1
        clear()
        env.render()
        print(f"Timestep: {epochs}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        sleep(0.15) # Sleep so the user can see the 

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {display_episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / display_episodes}")
print(f"Average penalties per episode: {total_penalties / display_episodes}")
