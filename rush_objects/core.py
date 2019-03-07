from collections import deque
from abc import ABC, abstractmethod

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

