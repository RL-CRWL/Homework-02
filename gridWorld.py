import numpy as np
import random

BOARD_ROWS = 7
BOARD_COLS = 7
INIT_STATE = (6, 0)  
GOAL_STATE = (0, 0)  
OBSTACLES = [(2, c) for c in range(6)]
ACTIONS = ["N", "S", "W", "E"]


class Gridworld:
    def __init__(self):
        self.state = INIT_STATE

    def step(self, action):
        r, c = self.state
        if action == "N":
            r -= 1
        elif action == "S":
            r += 1
        elif action == "W":
            c -= 1
        elif action == "E":
            c += 1

      
        if r < 0 or r >= BOARD_ROWS or c < 0 or c >= BOARD_COLS:
            r, c = self.state

        
        if (r, c) in OBSTACLES:
            r, c = self.state

        self.state = (r, c)

      
        if self.state == GOAL_STATE:
            return self.state, 20
        else:
            return self.state, -1

    def reset(self):
        self.state = INIT_STATE
        return self.state


def run_random_agent():
    env = Gridworld()
    env.reset()
    total_reward = 0
    for step in range(50):
        action = random.choice(ACTIONS)
        new_state, reward = env.step(action)
        total_reward += reward
       


def value_iteration():
    
    V = {(r, c): 0 for r in range(BOARD_ROWS) for c in range(BOARD_COLS) if (r, c) not in OBSTACLES}

    gamma = 1.0  # no discounting
    theta = 1e-4
    while True:
        delta = 0
        for state in V.keys():
            if state == GOAL_STATE:
                continue
            v = V[state]
            values = []
            for action in ACTIONS:
                env = Gridworld()
                env.state = state
                new_state, reward = env.step(action)
                values.append(reward + gamma * V.get(new_state, 0))
            V[state] = max(values)
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V


def run_greedy_agent(V):
    env = Gridworld()
    state = env.reset()
    total_reward = 0
    for step in range(50):
        # Choose action that maximizes value
        best_action = max(ACTIONS, key=lambda a: (
            env.step(a)[1] + V.get(env.step(a)[0], 0)
        ))
        state, reward = env.step(best_action)
        total_reward += reward
        if state == GOAL_STATE:
            break


if __name__ == "__main__":
    print("Running random agent:")
    run_random_agent()

    print("\nComputing optimal value function...")
    V_opt = value_iteration()
    for r in range(BOARD_ROWS):
        print([round(V_opt.get((r, c), None), 1) if (r, c) in V_opt else "0" for c in range(BOARD_COLS)])

    print("\nRunning greedy agent using optimal value function:")
    run_greedy_agent(V_opt)