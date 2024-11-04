from AlphaNetPlay import AlphaNetPlayer
from GameMeta import GameMeta
from GameState import GameState

"""
This is the Agent, which is able to play with the output of the engine
"""

"""
Initialize the AlphaNet Agent
Neural Network Name, Version (Iter) and Searches have to be set in the GameMeta file
:param player: The player who plays the game white or black (use the GameMeta class for choice)
"""
NName = GameMeta.COMP_NEURAL_NET_NAME
NIter = GameMeta.COMP_NEURAL_NET_ITER
NNSearches = GameMeta.COMP_MCTS_SEARCH_DEPTH
Agent1 = None
Agent2 = None
State = GameState(size=GameMeta.BOARD_SIZE, verbose=GameMeta.COMP_GAME_STATE_VERBOSE)
last_move_1 = None
last_move_2 = None


def agent(board: list[list[int]], action_set: list[tuple[int, int]]) -> tuple[int, int]:
    """
    Gets the action based on a board and action set. This method is tied down to not allow any errors
    Please report exceptions!

    Extracts the opponents move from the action list and plays it.
    Exception: If the player is to play first, detection through board check

    :param board: The current active board passed from the engine
    :param action_set: The available moves passed from the engine
    :return: The selected action to take
    """
    global Agent1
    global Agent2
    global State
    global last_move_1
    global last_move_2

    if Agent1 is not None and len(action_set) == 49:
        Agent1 = None
        State.reset()

    if Agent2 is not None and len(action_set) == 48:
        Agent2 = None

    if Agent1 is None:
        Agent1 = AlphaNetPlayer(NName, NIter, GameMeta.PLAYERS["white"], NNSearches, training_mode=GameMeta.COMP_TRAINING_MODE)

    if Agent2 is None:
        Agent2 = AlphaNetPlayer(NName, NIter, GameMeta.PLAYERS["black"], NNSearches, training_mode=GameMeta.COMP_TRAINING_MODE)

    if detect_player(len(action_set)) == GameMeta.PLAYERS["white"]:
        if len(action_set) == 49:
            action = Agent1.get_action(State)
            assert action in action_set
            State.move(action)
            last_move_1 = action
            return action

        known_action_space = State.get_action_space()

        if len(action_set) == len(known_action_space) - 1:
            other_player_move_list = list(set(known_action_space) - set(action_set))
            assert len(other_player_move_list) == 1
            other_player_move = other_player_move_list[0]
            State.move(other_player_move)
        else:
            other_player_move = last_move_2
        action = Agent1.get_action(State, other_player_move)
        assert action in action_set
        State.move(action)
        if State.winner != GameMeta.PLAYERS["none"]:  # Reset for new game
            Agent1 = None
            State.reset()
        last_move_1 = action
        return action

    elif detect_player(len(action_set)) == GameMeta.PLAYERS["black"]:
        known_action_space = State.get_action_space()

        if len(action_set) == len(known_action_space) - 1:
            other_player_move_list = list(set(known_action_space) - set(action_set))
            assert len(other_player_move_list) == 1
            other_player_move = other_player_move_list[0]
            State.move(other_player_move)
        else:
            other_player_move = last_move_1

        action = Agent2.get_action(State, other_player_move)
        assert action in action_set
        State.move(action)
        if State.winner != GameMeta.PLAYERS["none"]:  # Reset for new game
            Agent2 = None
            State.reset()
        last_move_2 = action
        return action


def detect_player(action_set_length: int) -> int:
    if action_set_length % 2 == 0:
        return GameMeta.PLAYERS["black"]
    else:
        return GameMeta.PLAYERS["white"]
