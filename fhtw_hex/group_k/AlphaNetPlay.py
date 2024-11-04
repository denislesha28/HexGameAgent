import os

import numpy as np
import torch

from GameMeta import GameMeta
from GameState import GameState
from MCTS import MCTS
from Networks import AlphaNet


class AlphaNetPlayer:
    """
    This class manages MCTS and Network, consequently also action selection. DO NOT MODIFY THIS
    """
    def __init__(self, neural_net_name: str, iteration: int, player: int = GameMeta.PLAYERS["white"], number_of_searches: int = 2000, training_mode: bool = True):
        """
        Initializes a Hex Player
        :param neural_net_name: The name of the neural network to be used for the MCTS
        :param iteration: Iteration of the neural network
        :param player: Player that plays the game
        :param number_of_searches: Search depth for the MCTS (not actual depth, number of visits)
        :param training_mode: Defines whether training mode is active or not
        """
        self.Network = self.load_network(neural_net_name, iteration)
        self.State = GameState(size=GameMeta.BOARD_SIZE)
        self.MCTS = None
        assert player is not GameMeta.PLAYERS["none"]
        self.Player = player
        self.NSearches = number_of_searches
        self.TrainingMode = training_mode

    def get_action(self, state: GameState, other_player_move: tuple[int, int] = None, decay_round: int = 0) -> tuple[int, int]:
        """
        Gets the action and updates the MCTS state

        "IF Section" at the beginning manages the state of the MCTS based on the other player move

        Performs an MCTS-Search with N number of visits, a possible winning move will always be selected

        Returns an action based on the probability distribution

        :param state: The current Game State
        :param other_player_move: The move the other played, so that root can be updated
        :param decay_round: Used for Rollout decay
        :return: The selected action
        """

        with torch.no_grad():
            actual_root_temp = False
            if self.MCTS is None:
                self.MCTS = MCTS(self.Network, training_mode=self.TrainingMode, decay_round=decay_round)
            if not self.MCTS.root.children:
                self.MCTS.expand_root(self.State, actual_root=True)
                if other_player_move is not None:
                    actual_root_temp = True
            if other_player_move is not None:
                self.MCTS.root = self.MCTS.root.children[other_player_move]
            if state not in self.MCTS.visited_boards:
                self.MCTS.expand_root(state, actual_root=actual_root_temp)
            winning_move = self.MCTS.search(self.NSearches, state)
            if winning_move is not None:
                if winning_move in state.get_action_space():
                    return winning_move
            search_probs = self.MCTS.get_search_probabilities(self.MCTS.root)
            moves = list(search_probs.keys())
            probs = list(search_probs.values())
            valid_move_selected = False
            while not valid_move_selected:
                chosen_move = np.random.choice(len(moves), p=probs)
                action = moves[chosen_move]
                if action in state.get_action_space():
                    valid_move_selected = True
            self.MCTS.root = self.MCTS.root.children[action]
            return action

    @staticmethod
    def load_network(neural_net_name, iteration):
        """
        Loads saved model and optimizer states if exists
        :param neural_net_name: Name of neural network
        :param iteration: Number of the iteration the neural network is from
        """
        net = AlphaNet()
        net.eval()
        base_path = "./group_k/model_data/"
        checkpoint_path = os.path.join(base_path, "%s_iter%d.pth.tar" % (neural_net_name, iteration))
        start_epoch, checkpoint = 0, None
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
        if checkpoint is not None:
            if len(checkpoint) == 1:
                net.load_state_dict(checkpoint['state_dict'])
            else:
                net.load_state_dict(checkpoint['state_dict'])

        if torch.cuda.is_available():
            net.cuda()
        return net
