# CrazyZero

The full implementation of the program.

## Installation

- the [CppFlow](https://github.com/serizba/cppflow) library depends directly on the Tensorflow C API to run TensorFlow models, so you should [download](https://www.tensorflow.org/install/lang_c) and install it before compiling (if an error occurs, try putting the `tensorflow.dll` file into the same directory as the executable)

- the saved neural network model should be placed into the directory `model` in the same directory as the executable

## Running self-play tests

- make sure that the file `self_play_configs.txt` is located in the same directory as the executable

- to change the testing parameters (which configurations and what numbers of simulations per move should be tested) change the contents of this file

- the first line defines the number of simulations per move (min-max-increment), while the other lines define the configurations to be tested

- to run the test add the `--self-play` flag when running the program
