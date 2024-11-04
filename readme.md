# Hex Game Agent 
- Hex Game Agent based on the AlphaZero-agent architecture
- The agent is trained from scratch using reinforcement learning

## Setup
- Utilise the requirements.txt to ensure the correct python packages are installed
- Run the example Script to test the agent, you may need to adapt the path that adds the modules in order for this to run
- The group_k module includes everything our agent needs to run

## Run
- from fhtw_hex.group_k.AlphaNetAgent import agent (Depending on the environment change this)
- Pass agent to the game.machine_vs_machine directly

## Troubleshooting
In case you see something like this:
```
  File "C:\projects\GroupK_HexAgent\example_script.py", line 15, in <module>
    from fhtw_hex.group_k.AlphaNetAgent import agent
  File "C:\projects\GroupK_HexAgent\fhtw_hex\group_k\AlphaNetAgent.py", line 1, in <module>
    from AlphaNetPlay import AlphaNetPlayer
ModuleNotFoundError: No module named 'AlphaNetPlay'
```

Change the include path for the module to a full path:
Example:
```
sys.path.append("C:/projects/GroupK_HexAgent/fhtw_hex/group_k/")
```

Depending from where the script is started it may be necessary to add multiple paths:
```
sys.path.append("C:/projects/GroupK_HexAgent/")
sys.path.append("C:/projects/GroupK_HexAgent/fhtw_hex/group_k/")
```