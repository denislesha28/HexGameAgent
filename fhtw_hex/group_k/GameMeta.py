"""
All variables and hyperparameters settings

Do not modify sections marked with [FIXED]
It is not advised to change the values in this section [SEMI-FIXED]
You may adapt the values in sections marked with [CUSTOM]
"""


class GameMeta:

    """
    Game Settings [FIXED]
    """
    PLAYERS = {'none': 0, 'white': 1, 'black': -1}
    GAME_OVER = -1
    # Change the board Size only for Debugging
    BOARD_SIZE = 7

    """
    Neural Network Settings [FIXED]
    """
    NN_INPUT_CHANNELS = 1
    NN_NUM_CHANNELS = 512
    NN_DROPOUT = 0.3
    NN_BOARD_SIZE = BOARD_SIZE
    NN_ACTION_SIZE = BOARD_SIZE**2
    NN_KERNEL_SIZE = 3
    NN_STRIDE_SIZE = 1
    NN_PADDING_SIZE = 1
    NN_MOMENTUM = 0.9
    NN_GAMMA = 0.77

    """
    Training Settings [CUSTOM]
    """
    # Iteration Start
    TRG_START_ITER = 0
    # Total Iterations
    TRG_TOTAL_ITER = 6
    # Iteration ID for the same network training
    TRG_ITER_ID = 77
    # Perform evaluation and retraining or not
    TRG_EVAL_AND_RETRAIN = True
    # Maximum Retraining Tries when Training
    TRG_MAX_RE_TRAIN = 3
    # Name of the Neural Network
    TRG_NEURAL_NET_NAME = "SkyNet"
    # Training Batch Size for the Dataloader
    TRG_BATCH_SIZE = 64
    # Epochs
    TRG_NUM_EPOCHS = 120
    # Learning Rate
    TRG_LRG_RATE = 0.2
    # Gradient Acceleration Steps
    TRG_GRAD_ACC_STEPS = 1
    # Max Norm
    TRG_MAX_NORM = 1.0
    # Number of process for Self Play
    TRG_NUM_PROCESS = 28
    # Number of games to play each Iteration per Process
    TRG_NUM_GAMES = 5
    # New Optim State
    TRG_NEW_OPTIM_STATE = True
    # Milestones at which the learning Rate is reduced
    TRG_MILESTONES = [30, 60, 90]
    # Number of Searches in the MCTS per Search Request
    TRG_MCTS_SEARCH_DEPTH = 800
    # Noise = 0 or Best Move Dict = 1, any else deactivates it
    TRG_ALT_MODE = 0

    """
    Evaluation Settings [CUSTOM]
    """
    # Number of games to play during evaluation
    EVAL_NUM_GAMES = 100
    # Win Percentage Threshold for a Network to beat another
    EVAL_WIN_THRESHOLD = 0.55
    # Number of Searches in the MCTS per Search Request
    EVAL_MCTS_SEARCH_DEPTH = 300
    # Verbosity of the Game State (Print)
    EVAL_GAME_STATE_VERBOSE = False

    """
    Evaluation Setting [FIXED]
    """
    EVAL_TRAINING_MODE = False

    """
    MCTS Settings [CUSTOM]
    """
    # Alpha Value for Dirichlet Noise
    MCTS_NOISE_ALPHA = 0.01
    # Epsilon Value for Dirichlet Noise
    MCTS_NOISE_EPSILON = 0.5
    # Rollout Percentage (set to 0.0 to deactivate)
    MCTS_ROLLOUT_PERCENTAGE = 0.0
    # Rollout Amount (must be larger than 0, even if it will not be used)
    MCTS_ROLLOUT_AMOUNT = 10
    # Rollout Percentage Decay
    MCTS_ROLLOUT_DECAY = 0.5

    """
    Competitive AlphaNet Settings [CUSTOM]
    """
    # Neural Network Name
    COMP_NEURAL_NET_NAME = "SkyNet"
    # Which Iteration of the Network to use
    COMP_NEURAL_NET_ITER = 2
    # Number of Searches in the MCTS per Search Request
    COMP_MCTS_SEARCH_DEPTH = 300

    """
    Competitive AlphaNet Settings [FIXED]
    """
    COMP_TRAINING_MODE = False
    COMP_GAME_STATE_VERBOSE = False


