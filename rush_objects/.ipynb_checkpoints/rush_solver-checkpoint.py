#######################################################
#  Based on: https://github.com/aimacode/aima-python  #
#######################################################
from rush_objects.cars import Car, Board
# from utils import memoize, PriorityQueue
from collections import OrderedDict, deque


class Problem(object):
    """
    The abstract class for a formal problem.
    """

    def __init__(self, initial, goal=None):
        """
        The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.
        """
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """
        Return the actions that can be executed in the given
        state.
        """
        raise NotImplementedError

    def result(self, state, action):
        """
        Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).
        """
        raise NotImplementedError

    def goal_test(self, state):
        """
        Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough.
        """
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """
        Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1.
        If the problem
        is such that the path doesn't matter, this function will only look at
        state2.
        If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """
        For optimization problems, each state has a value.
        Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError
# ______________________________________________________________________________


class Node:
    """
    A node in a search tree.
    Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.
    Alsstate us to this state, and
    thestates g) to reach the node.  Other functions
    maystatet_first_graph_search and astar_search for
    an state values are handled. You will not need to
    substate
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
        self.parent_list = []
        if parent:
            self.depth = parent.depth + 1
            self.parent_list = parent.parent_list + [self.state]

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        # This is for the case of equality in the f?
        # return self.state < node.state
        return True

    def expand(self, problem):
        """
        List the nodes reachable in one step from this node.
        """
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action,
                         problem.path_cost(self.path_cost, self.state,
                                           action, next_state))
        return next_node

    def solution(self):
        """
        Return the sequence of actions to go from the root to this node.
        """
        return [node.action for node in self.path()[1:]]

    def path(self):
        """
        Return a list of nodes forming the path from the root to this node.
        """
        node = self
        path_back = [node.state]
        for i in range(self.depth):
            node = node.parent
            path_back.append(node.state)
        return list(reversed(path_back))

    # We want  a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


class RushGame(Problem):
    """
    Adaptation of the Rush Hour problem to the A*
    search function
    """

    def __init__(self, board):
        """
        Args:
            board(Board): the state of the board that we want to solve.
                it will be overwritten
        """
        self.board = board
        self.board.car_lot = OrderedDict(self.board.car_lot)
        self.helper_board = Board(self.board.size, self.board.uscita)
        initial = board.get_state()
        print(type(board))
        Problem.__init__(self, initial, None)

    def actions(self, state):
        """
        State is a board.
        From there the agent can move to any of its neighbors
        that have not been visited yet.
        """
        self.helper_board.recreate_board(state)
        action_list = self.helper_board.possible_moves()
        return action_list

    def result(self, state, action):
        """
        The result of going to a neighbor is that the neighbor
        would be added to the list of visited.
        Take care that the action is an instance of Vertex
        So we need to extract its id
        """
        # A bit of care with tuples and lists
        self.helper_board.recreate_board(state)
        would_be_state = self.helper_board.wannabe_state(action[0], action[1])
        return would_be_state

    def h(self, node):
        """
        h function is not implemented
        """
        return 0

    def goal_test(self, state):
        """
        The goal is reached when the red car is at the exit
        """
        self.helper_board.recreate_board(state)
        red_car = self.helper_board.get_car('red')
        return red_car.position == [self.helper_board.uscita,
                                    self.helper_board.size-2]

    def solve(self, astar=False):
        """
        Execute the solution
        """
        if astar:
            raise NotImplementedError("The import of Priority Queue is needed")
            return astar_search(self)
        else:
            return breadth_first_graph_search(self)


def best_first_graph_search(problem, f):
    """
    Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned.
    """
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    i = 0
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
        i += 1
        if i % 1000 == 0:
            print('Checked already {} nodes'.format(i))
        max_depth = 500
        if node.depth == max_depth:
            print('Could not find solution within {} steps, sorry!'.format(
                max_depth))
            return False
    print('Solution could not be found within the limits imposed')
    return False


def astar_search(problem, h=None):
    """
    A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass.
    """
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))


def breadth_first_graph_search(problem):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = deque([node])
    explored = set()
    i = 0
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
        i += 1
        if i % 1000 == 0:
            print('Checked already {} nodes'.format(i))
        if node.depth == 500:
            print('Could not find solution within 100 steps, sorry!')
            return False
    print('Solution could not be found within the limits imposed')
    return False
