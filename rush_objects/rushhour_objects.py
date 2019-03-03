from colorama import Fore
from colorama import Style
from copy import deepcopy
from collections import OrderedDict
import numpy as np

# ------------------------- COLOR DEFINITIONS ----------------------- #
DEFAULT_COLORS = dict()
DEFAULT_COLORS["green"]         = Fore.GREEN
DEFAULT_COLORS["red"]           = Fore.RED
DEFAULT_COLORS["blue"]          = Fore.BLUE
DEFAULT_COLORS["yellow"]        = Fore.YELLOW
DEFAULT_COLORS["purple"]        = Fore.MAGENTA
DEFAULT_COLORS["cyan"]          = Fore.CYAN
DEFAULT_COLORS["light-green"]   = Fore.LIGHTGREEN_EX
DEFAULT_COLORS["light-blue"]    = Fore.LIGHTBLUE_EX
DEFAULT_COLORS["pink"]          = Fore.LIGHTRED_EX
DEFAULT_COLORS["orange"]        = Fore.LIGHTMAGENTA_EX
DEFAULT_COLORS["dark"]          = Fore.BLACK
DEFAULT_COLORS["another_blue"]  = Fore.BLUE
DEFAULT_COLORS["another_green"] = Fore.GREEN
DEFAULT_COLORS["grey"]          = Fore.LIGHTBLACK_EX
DEFAULT_COLORS["grey"]          = Fore.LIGHTBLACK_EX


class Car(object):
    """
    Class to represent each Car that exists in our Game. It offers
    a nice API to connect with the board and to manipulate changes 
    in the state (position) of the car. 

    Arguments:
    ----------
    - color: string, valid colorname
    - position: list or tuple of the form [x,y] 
    - length: int, length of the car 
    - orientation: string, orientation of the car "v"/"h" 

    Attributes (generated):
    -----------------------
    - color: the color 
    - orid : orientation identifier, 1 == horizontal, 0 == vertical
    - place: the indices 'slice'
    
    Properties (runtime-evaluation)
    -------------------------------
    - position: the 'position' of the car
    - length  : the length of the car
    - orientation: the orientation of the car
    - npindices: the slice that correspond to the position of the car
    - get_color: API to connect with Fore for rendering 
    
    NOTE: Color is completely obsolate and should be removed in the future
    """
    # Use of __slots__ instead of __dict__ reduces memory consumption 
    __slots__ = ["color","place","orid"]

    def __init__(self, color, position, length, orientation):
        self.color = color
        row,col = position 
        if orientation == "h":
            self.orid = 1
            self.place = [row,np.arange(col,col+length,dtype=int)]
        elif orientation == "v":
            self.orid = 0
            self.place = [np.arange(row,row+length,dtype=int),col]
        else:
            raise ValueError("Unrecognized orientation: %s"%orientation) 
    
    @property
    def position(self):
        """Returns the (starting-point)/position of the car"""
        if self.orid == 0: 
            return (self.place[0][0],self.place[1])
        else:
            return (self.place[0],self.place[1][0])
    
    @property
    def get_color(self):
        return DEFAULT_COLOR.get(self.color,False)  
    
    @property
    def length(self):
        """The length of the car"""
        return self.place[self.orid].shape[0] 
    
    @property
    def orientation(self):
        """The orientation of the car"""
        return "h" if self.orid else "v"

    @property
    def npindices(self):
        if self.orid == 0: 
            return tuple([slice(self.place[0][0],self.place[0][-1]+1,1),self.place[1]]) 
        else:
            return tuple([self.place[0],slice(self.place[1][0],self.place[1][-1]+1,1)])
        
    
    def astuple(self):
        """Attributes of the instance as tuple (hashable)"""
        return (self.color,self.position,self.length,self.orientation)
    
    def can_move(self, step, limit):
        """
        Routine to check if a car can move towards the defined 
        orientation, given a corresponding limit
        
        Arguments:
        ----------
        - step : int, the displacement of the car
        - limit: int, lower/upper boundary that must be satisfied
        
        Returns:
        --------
          boolean regarding the validity of the movement 
        """
        if step < 0: 
            return self.place[self.orid][0] + step >= limit
        else:
            return self.place[self.orid][-1] + step < limit 

    # --------------------- BUILD-IN OVERLOADS ----------------------- # 
        
    def __str__(self):
        """String representation of the instance (connected with logger??)"""
        return "Instance of Car(color={0},position={1},length={2},orientation={3})".format(
                self.color,self.position,self.length,self.orientation) 
    
    def __repr__(self):
        return self.__str__()
    
    def __iadd__(self, other):
        """Moves the car along the positive direction 'other' steps"""
        self.place[self.orid] += other; return self
    
    def __isub__(self, other):
        """Moves the car along the negative direction 'other' steps"""
        self.place[self.orid] -= other; return self

    @classmethod
    def from_slots(cls, color, place, orid):
        """
        Constructs an instance given the properties as declared in
        the '__slots__' attribute of the class 
        """
        x,y = place 
        if orid == 0:
            orientation = "v" 
            position = [x[0],y]
            length = x.shape[0]
        elif orid == 1:
            orientation = "h"
            position = [x,y[0]]
            length = y.shape[0]
        else:
            raise ValueError("Wrong specifiers. Cannot construct instanse") 

        return cls(color,position,length,orientation) 
            

class Board(object):
    """
    Class to represent a board of specific (square) size, contaning
    a set of Cars, upon its limits. 
    
    Arguments:
    ----------
    - size   : int, dimension of the square board
    - exitrow: int, row where the exit of the board is located 
    
    Attributes:
    -----------
    - view: np.ndarray, view of the board 
    - cars: OrderedDict of Cars 
    - exitrow: int, row where the exit of the board is located
    
    Properties:
    -----------
    - connected_states: tuple, the 'board-states' that can be 
                        reached upon a displacement of a any
                        car (if allowed) by +1,-1 
                        
    """
    # Use of __slots__ instead of __dict__ reduces memory consumption 
    __slots__ = ["view","cars","exitrow"] 
    
    def __init__(self, size, exitrow):
        self.view = np.zeros((size,size),dtype=int)
        self.cars = OrderedDict()
        self.exitrow = exitrow
        
    @property
    def empty_places(self):
        return np.count_nonzero(self.view) 
    
    def insert_car(self,color,position,length,orientation):
        """Routine to add a Car in the Board if the color has not been used"""
        car = Car(color,position,length,orientation)
        
        if not np.all(self.view[car.npindices]==0):
            raise ValueError("Unable to place the car\n%s"%car) 
        else:
            if self.cars.get(car.color,False) == False:
                self.cars[car.color] = car
                self.view[car.npindices] = 1
            else:
                raise ValueError("Car with color '%s' already inserted"%car.color)
    
    def get_state(self):
        """Routine to obtain a hashable form of the board"""
        cars = [car.astuple() for car in self.cars.values()]
        return tuple([self.view.shape[0],self.exitrow]+cars) 
        
    @property
    def connected_states(self):
        states = list() 
        for color in self.cars.keys():
            for displacement,boardlimit in [(-1,0),(1,self.view.shape[0])]:
                if self.cars[color].can_move(displacement,boardlimit):
                    # Move the car without changing the 'view'
                    self.cars[color] += displacement 
                    
                    # Check if the 'new' position creates conflicts
                    if np.any(self.view[self.cars[color].npindices]==0): 
                        states.append(self.get_state())    
                    
                    # Return the car to its original position 
                    self.cars[color] -= displacement
                    
        return tuple(states)
    
    @classmethod
    def from_state(cls, state):
        """Construct a 'Board' object from a given (hashed) state"""
        board = cls(state[0],state[1])
        for item in state[2:]: board.insert_car(*item) 
        return board
            
    def get_view(self):
        """Routine to return a copy of the 'view' of the board"""
        return self.view.copy() 
    
    def update_view(self):
        """Global update of the 'view' of the board"""
        self.view[:,:] = 0
        for car in self.cars.values():
            self.view[car.npindices] = 1
        if np.any(self.view>1): 
            raise ValueError("Recreation of the 'view' raised conflicts") 
    
    # --------------------- BUILD-IN OVERLOADS ----------------------- # 
        
    def __setitem__(self, color, displacement):
        """With this overload you can modify the position of each car and 
        automatically update the 'view' of the board (if allowed)"""
        limit = self.view.shape[0] if displacement > 0 else 0
        if self.cars[color].can_move(displacement,limit):
            temp = self.cars[color].npindices
            self.cars[color] += displacement 
            if np.any(self.view[self.cars[color].npindices]==0):
                self.view[temp] = 0
                self.view[self.cars[color].npindices] = 1
            else:
                self.cars[color] -= displacement
                raise ValueError("Unable to reset the position of the car: 'case-0'")  
        else:
            raise ValueError("Unable to reset the position of the car: 'case-1'")
        
    def __str__(self):
        ncars = len(self.cars.keys())
        return "Board with view\n%s\ncontaining %s cars"%(self.view,ncars)
    
    def __repr__(self):
        return self.__str__()