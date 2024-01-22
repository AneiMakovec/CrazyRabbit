# CrazyRabbit

The full implementation of the program.

## Installation

- the [CppFlow](https://github.com/serizba/cppflow) library depends directly on the Tensorflow C API to run TensorFlow models, so you should [download](https://www.tensorflow.org/install/lang_c) and install it before compiling (if an error occurs, try putting the `tensorflow.dll` file into the same directory as the executable)

- the saved neural network model should be placed into a directory named `model` in the same directory as the executable

## UCI options

- `UCI_Variant`: only supports crazyhouse
- `TimeControl`: `Default` - enables the time control system for timed games, `None` - disables the time control system
- `Simulations/Move`: how many MCTS simulations per move should the program use if `TimeControl` is set to `None`
- `BestMoveStrategy`: `Default` - use the AlphaZero best move selection strategy, `Q-value` - use the CrazyAra best move selection strategy
- `NodeExpansionStrategy`: `Default` - use the AlphaZero node expansion strategy when performing simulations, `Exploration` - use the CrazyAra node expansion strategy when performing simulations
- `BackpropStrategy`: `Default` - use the AlphaZero strategy to backpropagate the simulation results along the move tree, `SMA` - use the CrazyAra strategy to backpropagate the simulation results along the move tree
- `UseOpenings`: enables the use of the openings book
- `UseMateSearch`: enables the use of the search for forced mates
- `MateSearchMaxDepth`: limits the depth of the mate search
- `Eval_Material`: enables the use of the Material Advantage Value Correction  
- `Eval_PawnStructure`: enables the use of the Pawn Structure Value Correction
- `Eval_KingSafety`: enables the use of the King Safety Value Correction
- `Eval_PiecePlacement`: enables the use of the Piece Placement Value Correction
- `Eval_BoardControl`: enables the use of the Board Control Value Correction
- `PE_Dirichlet`: enables the use of the Dirichlet Policy Enhancement
- `PE_CheckingMoves`: enables the use of the Policy Enhancement of checking moves
- `PE_ForkingMoves`: enables the use of the Policy Enhancement of forking moves
- `PE_DroppingMoves`: enables the use of the Policy Enhancement of dropping moves
- `PE_CapturingMoves`: enables the use of the Policy Enhancement of capturing moves
