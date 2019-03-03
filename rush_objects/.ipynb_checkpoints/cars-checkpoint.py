from colorama import Fore
from colorama import Style
from copy import deepcopy
import numpy as np


mapping_color = {'green': Fore.GREEN, 'g': Fore.GREEN,
                  'red': Fore.RED, 'r': Fore.RED,
                  'blue': Fore.BLUE,
                  'yellow': Fore.YELLOW,
                  'purple': Fore.MAGENTA,
                  'cyan': Fore.CYAN,
                  'light-green': Fore.LIGHTGREEN_EX,
                  'light-blue': Fore.LIGHTBLUE_EX,
                  'pink': Fore.LIGHTRED_EX,
                  'orange': Fore.LIGHTMAGENTA_EX,
                  'dark': Fore.BLACK,
                  'another_blue': Fore.BLUE,
                  'another_green': Fore.GREEN,
                  'magenta': Fore.MAGENTA,
                  'grey': Fore.LIGHTBLACK_EX,
                  'gray': Fore.LIGHTBLACK_EX,
                  }


class Car():
    def __init__(self, col, l, orient, position):
        self.col = col
        self.len = l
        self.orient = orient
        if self.orient not in ['h', 'v']:
            raise KeyError(
                '{} orientation not recognized. Only h or v'.format(orient))
        self.moving_index = 1 if self.orient == 'h' else 0
        if isinstance(position, tuple):
            self.position = list(position)
        elif isinstance(position, list):
            self.position = position
        else:
            raise TypeError(
                "type {} not supported for position".format(type(position)))
        self.position_slice = self.generate_position_slice()
        self.set_color()

    def set_color(self):
        try:
            self.color_car = mapping_color[self.col]
        except:
            raise KeyError("Color {} not recognized".format(self.col))

    def generate_position_slice(self):
        p = self.position
        if self.orient == 'h':
            return (np.array([p[0] for i in range(self.len)]),
                    np.array([p[1]+i for i in range(self.len)])
                    )
        elif self.orient == 'v':
            return (np.array([p[0]+i for i in range(self.len)]),
                    np.array([p[1] for i in range(self.len)])
                    )
        else:
            raise KeyError('{} not valid car orientation'
                           .format(self.orient))



class Board():
    def __init__(self, size, uscita):
        self.size = size
        self.uscita = uscita
        self.car_lot = {}
        self.board = np.zeros((self.size, self.size))
        self.adding_order = []

    def __eq__(self, other):
        return (self.get_state() == other.get_state() and
                self.size == other.size and self.uscita == self.uscita)

    def reset_board(self):
        self.board = np.zeros((self.size, self.size))
        return True

    def get_car(self, col):
        return self.car_lot[col]

    def get_state(self):
        state = [(col, car.len, car.orient,
                  tuple(car.position))
                 for col, car in self.car_lot.items()]
        state = tuple(state)
        return state

    def recreate_board(self, state):
        self.reset_board()
        for item in state:
            self.place_car(*item)
        return self.board

    def place_car(self, col, l, orient, position, verbose=False):
        """
        Places a new car in the car lot if possible
        """
        next_car = Car(col, l, orient, position)
        # Check if the position is compatible with the present configuration
        if self.check_compatibility(next_car):
            self.car_lot[col] = next_car
            self.board = self.add_car_to_board(next_car)
            self.adding_order.append(col)
        else:
            if verbose:
                print('Adding the car was not possible for the constraints')
            return False

    def check_compatibility(self, next_car):
        position_slice = next_car.position_slice
        return self.perform_checks(position_slice)

    def check_move(self, position_slice):
        return self.perform_checks(position_slice)

    def perform_checks(self, position_slice, previous_position_slice=None):
        c1 = all([i in range(self.size) for i in position_slice[0]])
        c2 = all([i in range(self.size) for i in position_slice[1]])
        if not all([c1, c2]):
            return False
        if previous_position_slice is None:
            c3 = all(self.board[position_slice] == 0)
        else:
            cb = self.board.copy()
            cb[previous_position_slice] = 0
            c3 = all(cb[position_slice] == 0)
        return c3

    def add_car_to_board(self, next_car):
        # This function is not idempotent
        self.board[next_car.position_slice] = 1
        return self.board

    def is_legal_move(self, moving_car, d):
        """
        moving_car(Car): the car object to move
        direction(int): it has to be + or - 1
        """
        # Horrible code duplication here!
        # Same lines below!
        ind = moving_car.moving_index
        ps = moving_car.position_slice
        proposed_position = (ps[0] + (d if ind == 0 else 0),
                             ps[1] + (d if ind == 1 else 0))
        return self.perform_checks(proposed_position,
                                   previous_position_slice=ps)

    def wannabe_state(self, moving_car, d):
        wannabe_state = self.move_car(moving_car, d)
        self.undo_move(moving_car, d)
        return wannabe_state

    def possible_moves(self):
        moves = []
        for col, car in self.car_lot.items():
            for d in [+1, -1]:
                if self.is_legal_move(car, d):
                    moves.append([car, d])
        return moves

    def move_car(self, moving_car, d):
        present_state = self.get_state()
        ind = moving_car.moving_index
        ps = moving_car.position_slice
        proposed_position = (ps[0] + (d if ind == 0 else 0),
                             ps[1] + (d if ind == 1 else 0))
        if self.is_legal_move(moving_car, d):
            # If it is possible to move the car, then move it
            self.board[moving_car.position_slice] = 0
            self.car_lot[moving_car.col].position[ind] = \
                moving_car.position[ind]+d
            self.car_lot[moving_car.col].position_slice = proposed_position
            self.board[moving_car.position_slice] = 1
            wannabe_state = self.get_state()
            assert wannabe_state != present_state
            return wannabe_state
        else:
            return False

    def undo_move(self, moving_car, d):
        return self.move_car(moving_car, d*-1)

    def convert_board_to_string(self):
        string_board = [['0' for i in range(self.size)]
                        for j in range(self.size)]
        for col, car in self.car_lot.items():
            for i, j in zip(car.position_slice[0], car.position_slice[1]):
                string_board[i][j] = '\x1b[1m'+Style.BRIGHT + \
                    car.color_car+"X"+Style.RESET_ALL
        displines = ''
        for i, line in enumerate(string_board):
            displines += '|' + '|'.join(line)+'|'
            if i == self.uscita:
                displines += '=>'
            displines += '\n'
        return displines

    def render(self):
        print(self.convert_board_to_string())
        return True
