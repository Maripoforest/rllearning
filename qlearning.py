import numpy as np

# Define the environment
class Environment:
    def __init__(self):
        self.state = 0
        self.end_states = [3, 6]

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if self.state + action > 6:
            return self.state, 0, True
        self.state += action
        if self.state in self.end_states:
            return self.state, 10, True
        return self.state, -1, False

# Define the Q-Table
class QTable:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.table = {}
        for state in states:
            for action in actions:
                self.table[(state, action)] = 0.0

    def get(self, state, action):
        return self.table[(state, action)]

    def set(self, state, action, value):
        self.table[(state, action)] = value

# Define the Q-Learning algorithm
def q_learning(env, q_table, alpha, gamma, epsilon, episodes):
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.random() < epsilon:
                action = np.random.choice(q_table.actions)
            else:
                action = np.argmax([q_table.get(state, a) for a in q_table.actions])
            next_state, reward, done = env.step(action)
            q_value = q_table.get(state, action)
            next_q_value = np.max([q_table.get(next_state, a) for a in q_table.actions])
            q_table.set(state, action, q_value + alpha * (reward + gamma * next_q_value - q_value))
            state = next_state

# Run the algorithm
if __name__ == "__main__":
    env = Environment()
    q_table = QTable(range(7), [1, 2, 3])
    q_learning(env, q_table, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000)
    print(q_table.table)
