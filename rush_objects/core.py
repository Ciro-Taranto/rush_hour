from rush_objects.rushhour_objects import Board
from collections import deque
from abc import ABC, abstractmethod
from functools import partial 

class Problem(ABC):
    """
    The abstract class for a formal problem.

    Arguments:
    ----------
    initial: state
    goal   : state 
    """
    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    @abstractmethod
    def actions(self, state):
        """
        Return the actions that can be executed in the given state.
        """
        pass

    @abstractmethod
    def result(self, state, action):
        """
        Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).
        """
        pass

    def goal_test(self, state):
        """
        Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough.
        """
        if isinstance(self.goal, list):
            return state in self.goal
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """
        Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the 
        problem is such that the path doesn't matter, this function will 
        only look at state2. If the path does matter, it will consider c 
        and maybe state1 and action. 
        
        The default method costs 1 for every step in the path.
        """
        return c + 1

    def value(self, state):
        """
        For optimization problems, each state has a value; e.g. algorithms
        such as Hill-climbing try to maximize this value
        """
        raise NotImplementedError
# ______________________________________________________________________________


class Node(ABC):
    """
    A node in a search tree.
    Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node.
    Note that if a state is arrived at by two paths, then there are two nodes with
    the same state.

    Arguments:
    ----------
    - state: hashable, the hash of the state.
    - parent: hashable, the hash of the parent state
    - action: the action to reach this state from the parent state
    - path_cost: float, the path cost from the root to this state
    """

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """
        Create a search tree Node, derived from a parent by an action.
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        self.parent_list = list() 
        if parent:
            self.depth = parent.depth + 1
            self.parent_list = parent.parent_list + [self.state]

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        # TODO: how to order, if the hashable are not integers?
        # This is for the case of equality in the f?
        # return self.state < node.state
        return True

    def expand(self, problem):
        """
        List the nodes reachable in one step from this node.
        """       
        return (self.child_node(problem,action) 
                for action in problem.actions(self.state))


    def child_node(self, problem, action):
        """
        Given the action it constructs the next (child) node 
        """
        node = problem.result(self.state, action)
        cost = problem.path_cost(self.path_cost, self.state, action, node)
        return Node(node, self, action, cost) 

    def solution(self):
        """
        Return the sequence of actions to go from the root to this node.
        """
        return map(lambda x: x.action, self.path()[1:]) 

    def path(self):
        """
        Return a list of nodes forming the path from the root to this node.
        """
        last_node = self 
        traceback = [last_node.state]
        for index in range(self.depth):
            last_node = last_node.parent
            traceback.append(last_node.state)
        return list(reversed(traceback)) 

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

class BreadthFirstGraphSeach(object):
    """
    Alternative way to define a function
    """
    def __init__(self,problem):
        if not isinstance(problem,Problem):
            raise ValueError("Algorithm expects a instance of a Problem")
        self.problem = problem 
        self.root = problem.initial
        self.frontier = deque([Node(self.root)])
        self.explored = set() 
        self.iterations = 0

        # Check if the problem that has been provided is trivial
        if self.problem.goal_test(self.root):
            print("Initial state of the problem is already a solution") 

    def __call__(self,max_depth=500,printouts=100):
        while self.frontier:
            node = self.frontier.popleft()
            self.explored.add(node.state)

            for child in node.expand(self.problem):
                if child.state not in self.explored and child not in self.frontier:
                    if self.problem.goal_test(child.state):
                        print("Solution found in %s steps"%self.iterations)
                        return child
                    self.frontier.append(child)

            self.iterations += 1
            if self.iterations%printouts == 0:
                print("- Checked already {0} nodes".format(self.iterations))
            if node.depth == max_depth:
                print("Could not find solution within the 'depth' limit")
                return False

        print("Solution could not be found within the limits imposed")
        self.iterations = 0
        self.frontier = deque([Node(self.root)])
        self.explored = set() 
        return False


def breadth_first_graph_search(problem,max_depth=500,printouts=100):
    """
    TODO: WRITE DOCUMENTATION FOR THIS ALGORITHM 
    """
    root = Node(problem.initial)
    if problem.goal_test(root.state): return root

    frontier = deque([root])
    explored = set() 
    steps = 0

    while frontier:
        node = frontier.popleft()
        explored.add(node.state)

        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    print("Solution found in %s steps"%steps)
                    return child
                frontier.append(child)

        steps += 1
        if steps%printouts == 0:
            print("- Checked alread {0} nodes".format(steps))
        if node.depth == max_depth:
            print("Could not find solution within the 'depth' limit")
            return False

    print("Solution could not be found within the limits imposed")
    return False

class RushGame(Problem):
    """
    Class to play the game.
    Takes as input the board with the original arrangement of the cars,
    and allows to solve it with AIMA [artificial intelligence modern approach] methods.

    Arguments:
    ----------
    - board: Board or tuple(state)

    Keyword arguments:
    ------------------
    - target: string, the target car (default is 'red')

    """
    def __init__(self, board, **kwargs):
        if isinstance(board, Board):
            self.initial = Board.from_state(board.get_state())
        elif isinstance(board, tuple):
            self.initial = Board.from_state(board)
        else:
            raise ValueError("RushHour initialized expects 'Board'/'tuple'")

        self.target_car = kwargs.get("target", "red")
        if self.target_car not in self.initial.cars:
            raise ValueError("Target {} not found in the board".format(self.target_car))
        else:
            if self.initial[self.target_car].position[0] != self.initial.exitrow:
                raise ValueError("Target 'car' not placed in the 'exitrow'")
            elif self.initial[self.target_car].orientation != 'h':
                raise ValueError("Target car must be oriented horizontally")
            else:
                targetcol = self.initial.view.shape[0] - self.initial[self.target_car].length
                self.target = (self.initial.exitrow, targetcol)

        Problem.__init__(self, self.initial.get_state(), None)

    def actions(self, state):
        """
        Instantiate the abstract method of the Problem class.
        Given a state, returns the states that can be reached from this
        one.

        Arguments:
        ----------
        state: hashable, the hashable of the state from which the board can be recreated

        Returns:
        --------
        possible_moves:tuple, the list of possible state that can be reached.
            please note that in this case there is a perfect equivalence between
            action and state that can be reached with it.
        """
        return Board.from_state(state).connected_states

    def result(self, state, action):
        """
        The result of an action is entering in the state, i.e.m actions and states are equivalent

        Arguments:
        ----------
        state: hashable, The state from which the action started
        action: hashable, The result of going to a state is... being in that state

        Returns:
        action, the state to be reached
        """
        #assert state != action
        return action

    def h(self, node):
        """
        h is the heuristic function
        """
        raise NotImplementedError

    def goal_test(self, state):
        """
        Check if target car is in proper position
        """
        board = Board.from_state(state)
        return board.cars[self.target_car].position == self.target

    def solve(self, astar=False):
        """
        Execute the solution
        """
        if astar:
            raise NotImplementedError("A star search requires a heuristic function and PriorityQueue")
            return astar_search(self)
        else:
            return breadth_first_graph_search(self)

# Conceptually and notationally its more correct to 
# - Define the board
# - Defind the game (which is a Problem) by providing the board and the rules
# - Feed this game into a Solver 














