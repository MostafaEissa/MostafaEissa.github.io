---
title: 2048
categories: [Programming]
date: 2020-05-13
---
In this post we will try to implement the game 2048 as a console app in python!
<!--more-->

### Introduction

If you are not familiar with the [game 2048](https://en.wikipedia.org/wiki/2048_(video_game)). It is a sliding block puzzle game where you slide numbered tiles to combine them into a tile with the number 2048. You can checkout an online version [here](https://play2048.co/) from the original creator.
 
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
def create_board():
    board = np.array([
        [None,None,None,None],
        [None,None,None,None],
        [None,None,None,None],
        [None,None,None,None]
        ])
    return board
```

We also need an additional function that will pick and empty cell and put a value of 2 or 4 into it.

```python
def fill_random_cell(board):
    empty_slots = [(i,j) for i in range(board.shape[0]) for j in range(board.shape[1]) if board[i,j] == None]
    idx = np.random.randint(len(empty_slots))
    value = np.random.choice([2,4])
    board[empty_slots[idx]] = value
```

As such, in order to initialize the board we simply create an empty board and fill a random position on it.

```python
def new_game():
    # create a random empty board
    board = create_board()

    # fill an empty position with 2 or 4
    fill_random_cell(board)

    return board
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

###  Game Loop

What is left is some function to draw the board to the screen and correctly handle user input. Drawing can be done by a simple two nested for loops but we must be careful because cells can be blank or contain values up to 1024 so we need to make sure they all get maximum possible width needed.

```python
def draw(board):
    width, height = board.shape
    for i in range(height):
        for j in range(width):
            value = '    ' if board[i,j] is None else str(board[i,j]).center(4)
            print(value,  end='|')
        print()
        print('----|----|----|----')
```

We will also need to read and parse user input.  To make things simply we will assume users will not enter wrong input so that we do not have to handle input validation.

```python
def get_user_input():
    inp = input()
    if inp == 'w':
        return 'up'
    elif inp == 's':
        return 'down'
    elif inp == 'a':
        return 'left'
    elif inp == 'd':
        return 'right'
```

At this point, we need a loop to read the user input, apply the provided movement (left, right, up, down) and check the game state for game end.



```python
def game_loop():
    # create a new game 
    board = new_game()

    while True:
        # draw board on screen
        draw(board)

        # get user input 
        dir = get_user_input()

        # update game board
        board = move(board, dir)
        fill_random_cell(board)
        
        # check game state
        is_won, is_lost = is_game_won(board), is_game_lost(board)

        if is_won:
            print('You Won!')
            break
        elif is_lost:
            print('You Lost!)
            break
        else:
            continue
```

and that is all :).

