# Fiddler: An AI Music Composer

The goal of the project is to model music compositions by capturing temporal dependencies in classical music composi- tions and eventually generate novel music compositions from the learned model. We use approximately 24000 music com- positions transcribed in ABC notation. We implement count- based n-gram language model, use its results as a baseline for recurrent neural network based methods and assess their abil- ity to generate structurally coherent, human-pleasing music. We achieved test accuracy of 69.78% for char-RNN and 73% for seq2seq model with both methods generating valid music compositions.

Detailed experiments, methodlogies and results are discussed in `paper.pdf`.

## Install

In order to install `fiddler` run the following command:

    python setup.py install
    
 This will install `fiddler` as a command line tool.
 

 ## Commands

 ### Training Recurrent Neural Network
 
    fiddler train_rnn [options]

`train_rnn` command supports following options:

```
Usage: fiddler train_rnn [OPTIONS]
  Train neural network

Options:
  -f, --file PATH            Train Data File Path
  -b, --batch-size INTEGER   Batch size
  -l, --layers INTEGER       Number of layers in the network
  -r, --learning-rate FLOAT  Learning Rate
  -n, --num-steps INTEGER    No. of time steps in RNN
  -s, --cell-size INTEGER    Dimension of cell states
  -d, --dropout FLOAT        Dropout probability for the output
  -e, --epochs INTEGER       No. of epochs to run training for
  -c, --cell [lstm|gru]      Type of cell used in RNN
  -t, --test-seed TEXT       Seed input for printing predicted text after each
                             training step
  --delim / --no-delim       Delimit tunes with start and end symbol
  --save / --no-save         Save model to file
  --help                     Show this message and exit.
```
 
 If `fiddler` is not installed as command-line tool, you can use the same command using `python src/cli.py` with same arguments.

 ### Generate music using a trained RNN model

```
Usage: fiddler generate [OPTIONS]

Options:
  -m, --model_path PATH  Directory path for saved model
  --help                 Show this message and exit.
```

## Contributors

[Manthan Thakar](https://github.com/manthan787)   - Character-level Recurrent Neural Network and Sequence to Sequence Implementation

[Rashmi Dwarka](https://github.com/dwaraka-rashmi)    - Character-level Recurrent Neural Network and Sequence to Sequence Implementation

[Tirthraj Parmar](https://github.com/Tirthraj93)  - Character-level n-gram language model implementation
