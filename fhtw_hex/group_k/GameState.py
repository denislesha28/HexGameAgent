import copy
import numpy as np

from GameMeta import GameMeta


class GameState(object):
    """
    Essentially a copy of the hex_engine, but using numpy for the board and added hash, repr and eq functions
    """
    def __init__(self, size=7, verbose = False):
        """
        Initializes the GameState
        :param size: Sets the board size
        :param verbose: Sets if the board should be printed for every move made
        """
        # enforce lower and upper bound on size
        size = max(2, min(size, 26))
        # attributes encoding a game state
        self.size = max(2, min(size, 26))
        board = np.zeros((size, size), dtype=int)
        self.board = board
        self.player = GameMeta.PLAYERS["white"]
        self.winner = GameMeta.PLAYERS["none"]
        # attributes storing the history
        self.history = [board]
        self.verbose = verbose

    def reset(self):
        """
        This method resets the hex board. All stones are removed from the board and the history is cleared.
        """
        self.board = np.zeros((self.size, self.size))
        self.player = 1
        self.winner = 0
        self.history = []

    def move(self, coordinates):
        """
        This method enacts a move.
        The variable 'coordinates' is a tuple of board coordinates.
        The variable 'player_num' is either 1 (white) or -1 (black).
        """
        assert (self.winner == 0), "The game is already won."
        assert (self.board[coordinates[0], coordinates[1]] == 0), "These coordinates already contain a stone."
        from copy import deepcopy
        # make the moove
        self.board[coordinates[0], coordinates[1]] = self.player
        # change the active player
        self.player *= -1
        # evaluate the position
        self.evaluate(self.verbose)
        # append to history
        self.history.append(deepcopy(self.board))

        if self.verbose:
            self.print()

    def _get_adjacent(self, coordinates):
        """
        Helper function to obtain adjacent cells in the board array.
        Used in position evaluation to construct paths through the board.
        """
        u = (coordinates[0] - 1, coordinates[1])
        d = (coordinates[0] + 1, coordinates[1])
        r = (coordinates[0], coordinates[1] - 1)
        l = (coordinates[0], coordinates[1] + 1)
        ur = (coordinates[0] - 1, coordinates[1] + 1)
        dl = (coordinates[0] + 1, coordinates[1] - 1)
        return [pair for pair in [u, d, r, l, ur, dl] if
                max(pair[0], pair[1]) <= self.size - 1 and min(pair[0], pair[1]) >= 0]

    def get_action_space(self, recode_black_as_white=False):
        """
        This method returns a list of array positions which are empty (on which stones may be put).
        """
        actions = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    actions.append((i, j))
        if recode_black_as_white:
            return [self.recode_coordinates(action) for action in actions]
        else:
            return (actions)

    def _prolong_path(self, path):
        """
        A helper function used for board evaluation.
        """
        player = self.board[path[-1][0], path[-1][1]]
        candidates = self._get_adjacent(path[-1])
        # preclude loops
        candidates = [cand for cand in candidates if cand not in path]
        candidates = [cand for cand in candidates if self.board[cand[0]][cand[1]] == player]
        return [path + [cand] for cand in candidates]

    def evaluate(self, verbose=False):
        """
        Evaluates the board position and adjusts the 'winner' attribute of the object accordingly.
        """
        self._evaluate_white(verbose=verbose)
        self._evaluate_black(verbose=verbose)

    def _evaluate_white(self, verbose):
        """
        Evaluate whether the board position is a win for player '1'. Uses breadth first search.
        If verbose=True a winning path will be printed to the standard output (if one exists).
        This method may be time-consuming for huge board sizes.
        """
        paths = []
        visited = []
        for i in range(self.size):
            if self.board[i, 0] == 1:
                paths.append([(i, 0)])
                visited.append([(i, 0)])
        while True:
            if len(paths) == 0:
                return False
            for path in paths:
                prolongations = self._prolong_path(path)
                paths.remove(path)
                for new in prolongations:
                    if new[-1][1] == self.size - 1:
                        if verbose:
                            print("A winning path for 'white' ('1'):\n", new)
                        self.winner = 1
                        return True
                    if new[-1] not in visited:
                        paths.append(new)
                        visited.append(new[-1])

    def _evaluate_black(self, verbose):
        """
        Evaluate whether the board position is a win for player '-1'. Uses breadth first search.
        If verbose=True a winning path will be printed to the standard output (if one exists).
        This method may be time-consuming for huge board sizes.
        """
        paths = []
        visited = []
        for i in range(self.size):
            if self.board[0, i] == -1:
                paths.append([(0, i)])
                visited.append([(0, i)])
        while True:
            if len(paths) == 0:
                return False
            for path in paths:
                prolongations = self._prolong_path(path)
                paths.remove(path)
                for new in prolongations:
                    if new[-1][0] == self.size - 1:
                        if verbose:
                            print("A winning path for 'black' ('-1'):\n", new)
                        self.winner = -1
                        return True
                    if new[-1] not in visited:
                        paths.append(new)
                        visited.append(new[-1])

    def recode_black_as_white(self, input_board = None):
        """
        Returns a board where black is recoded as white and wants to connect horizontally.
        This corresponds to flipping the board along the south-west to north-east diagonal and swapping colors.
        This may be used to train AI players in a 'color-blind' way.
        """
        board = copy.deepcopy(self.board)
        if input_board is not None:
            board = input_board
        return board.T * -1

    def recode_coordinates(self, coordinates: tuple[int, int]):
        """
        Transforms a coordinate tuple (with respect to the board) analogously to the method recode_black_as_white.
        """
        assert (0 <= coordinates[0] and self.size - 1 >= coordinates[
            0]), "There is something wrong with the first coordinate."
        assert (0 <= coordinates[1] and self.size - 1 >= coordinates[
            1]), "There is something wrong with the second coordinate."
        return (self.size - 1 - coordinates[1], self.size - 1 - coordinates[0])

    def coordinate_to_scalar(self, coordinates):
        """
        Helper function to convert coordinates to scalars.
        This may be used as alternative coding for the action space.
        """
        assert (0 <= coordinates[0] and self.size - 1 >= coordinates[
            0]), "There is something wrong with the first coordinate."
        assert (0 <= coordinates[1] and self.size - 1 >= coordinates[
            1]), "There is something wrong with the second coordinate."
        return coordinates[0] * self.size + coordinates[1]

    def scalar_to_coordinates(self, scalar):
        """
        Helper function to transform a scalar "back" to coordinates.
        Reverses the output of 'coordinate_to_scalar'.
        """
        coord1 = int(scalar / self.size)
        coord2 = scalar - coord1 * self.size
        assert (0 <= coord1 and self.size - 1 >= coord1), "The scalar input is invalid."
        assert (0 <= coord2 and self.size - 1 >= coord2), "The scalar input is invalid."
        return (coord1, coord2)

    def print(self, invert_colors=True):
        """
        This method prints a visualization of the hex board to the standard output.
        If the standard output prints black text on a white background, one must set invert_colors=False.
        """
        names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        indent = 0
        headings = " " * 5 + (" " * 3).join(names[:self.size])
        print(headings)
        tops = " " * 5 + (" " * 3).join("_" * self.size)
        print(tops)
        roof = " " * 4 + "/ \\" + "_/ \\" * (self.size - 1)
        print(roof)
        # color mapping inverted by default for display in terminal.
        if invert_colors:
            color_mapping = lambda i: " " if i == 0 else ("\u25CB" if i == -1 else "\u25CF")
        else:
            color_mapping = lambda i: " " if i == 0 else ("\u25CF" if i == -1 else "\u25CB")
        for r in range(self.size):
            row_mid = " " * indent
            row_mid += "   | "
            row_mid += " | ".join(map(color_mapping, self.board[r]))
            row_mid += " | {} ".format(r + 1)
            print(row_mid)
            row_bottom = " " * indent
            row_bottom += " " * 3 + " \\_/" * self.size
            if r < self.size - 1:
                row_bottom += " \\"
            print(row_bottom)
            indent += 2
        headings = " " * (indent - 2) + headings
        print(headings)

    def __hash__(self):
        return hash(str(self.board))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __repr__(self):
        return str(self.board)

