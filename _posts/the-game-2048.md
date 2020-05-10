---
layout: article
title: Value Rigidity
tags: [Programming]
---
In this post we will try to implement the game 2048 as a console app in python!
<!--more-->

### Introduction

If you are not familiar with the [game 2048](https://en.wikipedia.org/wiki/2048_(video_game)). It is a sliding block puzzle game where you slide numbered tiles to combine them into a tile with the number 2048. You can checkout an online version [here] (https://play2048.co/) from the original creator.
 
### Game Rules

In order to implement the game we need to understand its rules. The game has only a few rules:

1. At every turn, a tile randomly appears on the board with a value of 2 or 4.
2. The player can move the tiles in one of the four direction {left, right, up, down}

    - tiles slides as far as possible in the played direction until stopped by another tile or the board edge 
    - when two tiles with the same number collide together, they are merged ino a new title whose value is twice the value of the individual tile 
    - a merged tile cannot merge with another tile in the same move.

### Game State

We can represent the game as a 2D array which each cell position is either empty or contains a value that is a multiple of 2.

```python
board = np.array([
    [None,None,None,None],
    [None,None,None,None],
    [None,None,None,None],
    [None,None,None,None]
    ])
```

We also need an additional function that will pick and empty cell and put a value of 2 or 4 into it.

```python
def fill_random_cell(board):

```

### Merging of Tiles 

We need a couple of helper functions that can help us maage the game state. 

First, we need a function to move the tiles in a certain direction by applying the rules specified below. 

```python
def shift_left(board):
    width, height = board.shape
    for i in range(height):
        for k in range(width):
            for j in range(1, width):
                if board[i][j - 1] is None:
                    board[i][j - 1] = board[i][j]
                    board[i][j] = None

def merge_tiles(board):
    width, height = board.shape
    for i in range(height):
        for j in range(1, width):
            if board[i][j - 1] == board[i][j] and board[i][j] is not None:
                board[i][j - 1] *= 2
                board[i][j] = None
 ```

we can use the above functions to generalize board movement in all directions.         

```python
def move(board, direction):
    new_board = board.copy()
    directions = {
        'left': lambda x: x,
        'right': lambda board: np.flip(board),
        'up' : lambda board: board.T,
        'down':lambda board: np.flip(board).T
    }
    transform = directions[direction]
    
    new_board = transform(new_board)
    shift_left(new_board)
    merge_tiles(new_board)
    shift_left(new_board)
    new_board = transform(new_board)
    return new_board
```

### End of Game

Second, we need a function to determine if we reached the end of a game i.e. if a player has won if a tile has the value ```1024``` or lost if there are no more available moves left.

```python
def is_game_won(board):
    return (board == 1024).any()

def is_game_lost(board):
    # check all move direction
    # if no new board can be generated
    # then the game is lost
    if (move(board, 'left') == board).all() \
    and (move(board, 'right') == board).all() \
    and (move(board, 'up') == board).all() \
    and (move(board, 'down') == board).all() :
        return True
    
    return False
```

### Main Game Loop

Hope you enjoyed this post.
<br/>
M.

