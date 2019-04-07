from collections import deque
from rush_objects.core import Problem,Node

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
