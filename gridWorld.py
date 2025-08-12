import random
import matplotlib.pyplot as plt
import numpy as np


class Gridworld:
    def __init__(self):
        self.rows, self.cols = 7, 7
        self.start = (6, 0)  
        self.goal = (0, 0)  
        self.obstacles = [(2, c) for c in range(6)] #excludes right most cell
        self.state = self.start
        self.actions = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)}

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        dr, dc = self.actions[action]
        r, c = self.state
        nr, nc = r + dr, c + dc

        if (nr, nc) in self.obstacles or not (0 <= nr < self.rows and 0 <= nc < self.cols):
            nr, nc = r, c  # can't move

        self.state = (nr, nc)

        if self.state == self.goal:
            return self.state, 20, True
        else:
            return self.state, -1, False

def compute_optimal_value():
    V = np.zeros((7, 7))
    gamma = 1.0
    actions = [(-1,0), (1,0), (0,-1), (0,1)]
    obstacles = [(2, c) for c in range(6)]
    goal = (0,0)
    changed = True

    while changed:
        changed = False
        newV = np.copy(V)
        for r in range(7):
            for c in range(7):
                if (r,c) in obstacles or (r,c) == goal:
                    continue
                values = []
                for dr,dc in actions:
                    nr, nc = r+dr, c+dc
                    if (nr, nc) in obstacles or not (0 <= nr < 7 and 0 <= nc < 7):
                        nr, nc = r, c
                    reward = 20 if (nr, nc) == goal else -1
                    values.append(reward + gamma * V[nr, nc])
                best = max(values)
                if abs(best - V[r,c]) > 1e-6:
                    newV[r,c] = best
                    changed = True
        V = newV
    return V


def random_agent(env, steps=50):
    total_reward = 0
    trajectory = [env.state]
    for _ in range(steps):
        action = random.choice(list(env.actions.keys()))
        _, reward, done = env.step(action)
        total_reward += reward
        trajectory.append(env.state)
        if done:
            break
    return total_reward, trajectory

def greedy_agent(env, V, steps=50):
    total_reward = 0
    trajectory = [env.state]
    actions = list(env.actions.keys())
    for _ in range(steps):
        best_action = None
        best_value = float('-inf')
        for a in actions:
            dr, dc = env.actions[a]
            nr, nc = env.state[0] + dr, env.state[1] + dc
            if (nr, nc) in env.obstacles or not (0 <= nr < env.rows and 0 <= nc < env.cols):
                nr, nc = env.state
            reward = 20 if (nr, nc) == env.goal else -1
            value = reward + V[nr, nc]
            if value > best_value:
                best_value = value
                best_action = a
        _, reward, done = env.step(best_action)
        total_reward += reward
        trajectory.append(env.state)
        if done:
            break
    return total_reward, trajectory


env = Gridworld()
V = compute_optimal_value()

# 1. Bar graph of average returns
random_returns = []
greedy_returns = []
for _ in range(20):
    env.reset()
    r_return, _ = random_agent(env)
    random_returns.append(r_return)

    env.reset()
    g_return, _ = greedy_agent(env, V)
    greedy_returns.append(g_return)

avg_random = np.mean(random_returns)
avg_greedy = np.mean(greedy_returns)

plt.figure(figsize=(5,4))
plt.bar(['Random', 'Greedy'], [avg_random, avg_greedy], color=['red','green'])
plt.ylabel('Average Return over 20 runs')
plt.title('Agent Performance')
plt.show()
env.reset()
_, rand_traj = random_agent(env)
env.reset()
_, greedy_traj = greedy_agent(env, V)

def plot_trajectory(traj, title):
    grid = np.zeros((7,7))
    for i, (r,c) in enumerate(traj):
        grid[r,c] = i+1
    plt.imshow(grid, cmap='Blues', origin='upper')
    plt.title(title)
    for (r,c) in traj:
        plt.text(c, r, '•', ha='center', va='center', color='red', fontsize=12)
    plt.gca().invert_yaxis()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plot_trajectory(rand_traj, "Random Agent")
plt.subplot(1,2,2)
plot_trajectory(greedy_traj, "Greedy Agent")
plt.show()
