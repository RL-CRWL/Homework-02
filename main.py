import numpy as np

V: dict[tuple[int, int], int] = {}

for i in range(7):
    for j in range(7):
        V[(i,j)] = 0
        
moves = [(-1,0),# N
         (0,+1),# E
         (+1,0),# S
         (0,-1)]# W

def get_possible_moves(state):
    possible_moves = []
    
    for move in moves:
        x=state[0]+move[0] 
        y=state[1]+move[1]
        if (x<7 and x>=0) and (y<7 and y>=0):
            if (x==2 and y>5):
                possible_moves.append(move)
                continue
            elif (x==2 and y<=5):
                continue
            possible_moves.append(move)
    
    return possible_moves

def make_move(state, move):
    return (state[0]+move[0], state[1]+move[1])

visited:dict[tuple[int,int],list[tuple[int,int]]] = {}
def optimal_value_function(state):
    if state == (0,0):
        V[state] = 20 - 1
        return V[state]
    else:
        possible_moves = get_possible_moves(state)
        for move in possible_moves:
            if visited.get(state)==None:
                visited[state] = [move]
                V[state] = optimal_value_function(make_move(state, move)) - 1
            elif move in visited.get(state):
                continue
            else:
                visited[state].append(move)
                V[state] = optimal_value_function(make_move(state, move)) - 1
        return V[state]    

V[(6,0)] = optimal_value_function((6,0))

print(V)
    