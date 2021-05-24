# Taxi-v3 Q-Learning
 A simple Q-learning implementation in OpenAI Gym's "Taxi-v3" environment.
 
## What is OpenAI Gym?
[OpenAI Gym](https://gym.openai.com/) is a toolkit for developing and comparing reinforcement algorithms. It provides a wide range of environments with different reinforcement learning tasks.

It can be found on GitHub [here](https://github.com/openai/gym) and documentation is [here.](https://gym.openai.com/docs)

## Setup & Running the code.
Python 3 is required and can be downloaded [here.](https://www.python.org/downloads/)
### Installing required libraries.
```
pip3 install -r requirements.txt
```
### Running the agent.
```
py agent.py
```

## Possible Improvements
- Command line arguments to modify the amount of training episodes.
- Saving and loading the q-table.
- Tuning alpha, gamma and epsilon by decaying over training episodes.
