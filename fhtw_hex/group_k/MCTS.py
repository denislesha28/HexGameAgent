import math
import random
from copy import deepcopy
from queue import Queue
from random import choice
from time import time

import numpy as np
import torch

from GameMeta import GameMeta
from GameState import GameState
from Networks import AlphaNet
from Node import Node


best_move_dict_w = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 1), (1, 2), (1, 3)]
best_move_dict_b = [(3, 3), (3, 4), (4, 3), (4, 4)]

class MCTS:
    """
    MCTS for the Hex Game
    """

    def __init__(self, model: AlphaNet, use_value: bool = True, use_policy: bool = True, training_mode: bool = True,  verbose: bool = False, decay_round: int = 0):
        """
        Initialize the MCTS
        :param model: The network used to predict value and probabilities
        :param use_value: Toggle Value for certain cases
        :param use_policy: Use Network Policy
        :param training_mode: Training Mode active (influences noise)
        :param verbose: Passed to the GameState, activates prints
        """
        self.state = GameState(verbose)
        self.root = Node()
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0
        self.verbose = verbose

        self.use_policy = use_policy
        self.use_value = use_value
        self.model = model
        self.visited_boards = {}

        self.training_mode = training_mode
        self.alpha = GameMeta.MCTS_NOISE_ALPHA
        self.epsilon = GameMeta.MCTS_NOISE_EPSILON

        self.decay_round = decay_round

    def select_node(self) -> tuple[Node, GameState]:
        """
        Select the node with the largest UCB Weight and expand it
        :return: Node and updated GameState
        """

        node = self.root
        state = deepcopy(self.state)

        while len(node.children) != 0:
            children = node.children
            items = children.items()
            ucb_weights = [(v.ucb_weight(state.player), v) for k, v in items]
            tmp_node = max(ucb_weights, key=lambda c: c[0])
            if tmp_node[1].move in state.get_action_space():
                state.move(tmp_node[1].move)
                node = tmp_node[1]
            else:
                raise Exception("Not a valid move")

        self.expand(node, state)
        return node, state

    def expand(self, parent: Node, state: GameState) -> bool:
        """
        Calculate Node Value and Probabilities
        Add children when possible
        :param parent: The current Node
        :param state: The current Game State
        :return: Whether the Node was expanded (children were added)
        """
        value = 0
        if state.winner != GameMeta.PLAYERS['none']:
            value = 1 if state.winner == 1 else -1
            parent.value = value
            return False

        if self.training_mode is True and GameMeta.MCTS_ROLLOUT_PERCENTAGE > 0.0 and random.randint(1, math.ceil(1/(GameMeta.MCTS_ROLLOUT_PERCENTAGE * (GameMeta.MCTS_ROLLOUT_DECAY**self.decay_round)))) == 1:
            self.use_value = False

        if self.use_policy or self.use_value:
            probs, value = self.model_predict(state)
            parent.prior_policy = probs
        if not self.use_value:
            value = 0
            for i in range(0, GameMeta.MCTS_ROLLOUT_AMOUNT):
                value += self.roll_out(state)
            value = value / GameMeta.MCTS_ROLLOUT_AMOUNT

        self.use_value = True

        parent.value = value

        children = []

        valid_actions = state.get_action_space()

        for move in valid_actions:
            children.append(Node(move, parent, parent.prior_policy[move[0]][move[1]]))

        parent.set_children(children)
        self.visited_boards[state] = parent

        return True

    def expand_root(self, state: GameState, actual_root: bool = False) -> None:
        """
        Expands the current root and adds children
        Adds noise or probabilities for the best move to the very beginning of the game
        :param state: The current GameState
        :param actual_root: Determines if this is actually the start of a game
        """
        value = 0
        if self.use_policy or self.use_value:
            probs, value = self.model_predict(state)
            if self.training_mode and actual_root:
                if GameMeta.TRG_ALT_MODE == 0:
                    flat_priors = probs.flatten()
                    noise = np.random.dirichlet([self.alpha] * len(flat_priors))
                    noisy_priors = (flat_priors * (1 - self.epsilon) + noise * self.epsilon).reshape(probs.shape)
                    probs = noisy_priors
                elif GameMeta.TRG_ALT_MODE == 1:
                    probs = np.zeros((GameMeta.BOARD_SIZE, GameMeta.BOARD_SIZE))
                    if state.player == GameMeta.PLAYERS["white"]:
                        for move in best_move_dict_w:
                            probs[move[0]][move[1]] = 1 / len(best_move_dict_w)
                    else:
                        for move in best_move_dict_b:
                            probs[move[0]][move[1]] = 1 / len(best_move_dict_b)

            self.root.prior_policy = probs
        if not self.use_value:
            value = self.roll_out(state)

        self.root.value = value

        children = []

        valid_actions = state.get_action_space()

        for move in valid_actions:
            children.append(Node(move, self.root, self.root.prior_policy[move[0]][move[1]]))

        self.root.set_children(children)
        self.visited_boards[state] = self.root

    @staticmethod
    def roll_out(state: GameState) -> int:
        """
        Simulate an entirely random game from the passed state and return the winning player
        :param state: Current GameState
        :return: The winner of the simulated game
        """

        tmp_state = deepcopy(state)
        moves = tmp_state.get_action_space()

        while tmp_state.winner == GameMeta.PLAYERS['none']:
            move = choice(moves)
            tmp_state.move(move)
            moves.remove(move)

        return tmp_state.winner

    @staticmethod
    def back_propagation(node: Node, outcome: float) -> None:
        """
        Update Node and parents with the outcome
        :param node: The node to propagate back from
        :param outcome: The value for the node
        """
        while node is not None:
            node.update_value(outcome)
            node = node.parent

    def search(self, number_of_searches: int, current_state: GameState = None) -> tuple[int, int]:
        """
        Performs search in the MCTS and back propagates values through the tree
        :param number_of_searches: The amount of searches to perform
        :param current_state: The current Game State
        :return: Only returns if a winning move was found, if not always None
        """
        winning_move = self.check_for_winning_move(current_state)

        if winning_move is not None:
            return winning_move

        if current_state is not None:
            self.set_game_state(current_state)

        start_time = time()
        number_of_rollouts = 0

        while number_of_rollouts < number_of_searches:
            node, state = self.select_node()
            outcome = node.value
            self.back_propagation(node, outcome)
            number_of_rollouts += 1

        run_time = time() - start_time
        node_count = self.tree_size()
        self.run_time = run_time
        self.node_count = node_count
        self.num_rollouts = number_of_rollouts
        return None

    def move(self, move: tuple[int, int]) -> None:
        """
        Performs a move in the Game State, used for actual playing
        :param move: Coordinates of the move
        """
        if move in self.root.children:
            child = self.root.children[move]
            child.parent = None
            self.root = child
            self.state.move(child.move)
            return

        self.state.move(move)
        self.root = Node()

    def set_game_state(self, state: GameState, reset_node: bool = False) -> None:
        """
        Set the root_state of the tree to the passed Game State, this clears all
        the information stored in the tree since none of it applies to the new
        state.
        :param state: The Game State to set the MCTS State to
        :param reset_node: Resets the root node to an empty node
        """
        self.state = deepcopy(state)
        self.state.verbose = self.verbose
        if reset_node is True:
            self.root = Node()

    def statistics(self) -> tuple[int, int, int]:
        """
        Returns the statistics about the tree
        """

        return self.num_rollouts, self.node_count, self.run_time

    def tree_size(self) -> int:
        """
        Count nodes in tree by BFS.
        """
        Q = Queue()
        count = 0
        Q.put(self.root)
        while not Q.empty():
            node = Q.get()
            count += 1

            for child in node.children.values():
                Q.put(child)

        return count

    def model_predict(self, state: GameState) -> tuple:
        """
        Gets the Network prediction for value and probabilities of a Game State
        :param state: The current Game State
        :return: Tuple of probabilities and Value
        """
        encoded_board = state.board if state.player == GameMeta.PLAYERS['white'] else state.recode_black_as_white()
        board_tensor = torch.tensor(encoded_board, dtype=torch.float32)
        board_tensor = board_tensor.view(1, 1, GameMeta.BOARD_SIZE, GameMeta.BOARD_SIZE)
        if torch.cuda.is_available():
            board_tensor = board_tensor.cuda()
        probs, value = self.model(board_tensor)
        probs = probs.cpu().numpy().reshape((GameMeta.BOARD_SIZE, GameMeta.BOARD_SIZE))
        value = value.item()

        if state.player == GameMeta.PLAYERS['black']:
            probs = probs.T

        return probs, value

    def get_search_probabilities(self, node: Node) -> dict:
        """
        Returns the Probabilities for a Node's children based on their visits
        Winning moves have adapted Probabilities
        :param node: The node to get the probabilities from
        :return: A dictionary of moves and probabilities
        """
        children = node.children
        if node.has_winning_move:
            number_of_winning_moves = len(node.winning_moves)
            probs = {action: 1 / number_of_winning_moves if action in node.winning_moves else 0 for action, child in children.items()}
            node.has_winning_move = False
            node.winning_moves = []
            return probs

        items = children.items()
        child_visits = [child.visits for action, child in items]
        sum_visits = sum(child_visits)
        if sum_visits != 0:
            normalized_probs = {action: (child.visits/sum_visits) for action, child in items}
        else:
            normalized_probs = {action: (child.visits/len(child_visits)) for action, child in items}
        return normalized_probs

    def check_for_winning_move(self, state: GameState) -> tuple[int, int]:
        """
        Checks if any of the direct children are a winning move and adds them to the Node
        :param state: Current Game State
        :return: A randomly chosen Winning Move, None if no Winning Move was found
        """
        winning_moves = []
        for action in self.root.children:
            temp_state = deepcopy(state)
            temp_state.move(action)
            if temp_state.winner != GameMeta.PLAYERS["none"]:
                winning_moves.append(action)

        if len(winning_moves) > 0:
            self.root.has_winning_move = True
            self.root.winning_moves = winning_moves
            return choice(winning_moves)
        return None