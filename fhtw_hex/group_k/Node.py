import numpy as np

from GameMeta import GameMeta


class Node:
    """
    Node for the MCTS.
    """
    def __init__(self, move: tuple = None, parent: object = None, prior_prob: float = 1.0):
        """
        Initialize a new Node.

        :param move: The move taken to get to this position
        :param parent: The parent of this node, default = None
        :param prior_prob: The probability of being picked from the parent, when the parent was expanded
        """
        self.move = move
        self.parent = parent
        self.children = {}
        self.prior_policy = np.zeros((GameMeta.BOARD_SIZE, GameMeta.BOARD_SIZE))
        self.visits = 0
        self.value = 0.5
        self.prior_prob = prior_prob
        self.UCBConstant = 2
        self.has_winning_move = False
        self.winning_moves = []

    def set_children(self, children: list) -> None:
        """
        Sets the children of this node
        :param children: The children of this node
        """
        for child in children:
            self.children[child.move] = child

    def update_value(self, outcome: float):
        """
        Updates the Node Value
        :param outcome: The value to update the node with
        """
        self.value = (self.visits * self.value + outcome) / (self.visits + 1)
        self.visits += 1

    def ucb_weight(self, current_player: int) -> float:
        """
        The UCB Weight of a Node
        As the nodes visits increase the importance of the nodes value increases, if the node is not favorable, another will probably be chosen instead
        :param current_player: The player of the current Turn
        :return: The calculated UCB Weight
        """
        if current_player == GameMeta.PLAYERS["black"]:
            return -1 * self.value + self.UCBConstant*self.prior_prob/(1+self.visits)
        else:
            return self.value + self.UCBConstant*self.prior_prob/(1+self.visits)

    def __str__(self):
        return "Move: " + str(self.move) + " Value: " + str(self.value) + " Visits: " + str(self.visits)
