# fiddler
An AI Irish music composer


## Install

In order to install `fiddler` run the following command:

    python setup.py install
    
 This will install `fiddler` as a command line tool.
 

 # Commands

 ## Training Recurrent Neural Network
 
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
  -e, --epochs INTEGER       No. of epochs to run training for
  -c, --cell [lstm|gru]      Type of cell used in RNN
  -t, --test-seed TEXT       Seed input for printing predicted text after each
                             training step
  --delim / --no-delim       Delimit tunes with start and end symbol
  --help                     Show this message and exit.
```
 
 If `fiddler` is not installed as command-line tool, you can use the same command using `python src/cli.py` with same arguments.

 ## Generate music using a trained RNN model

     fiddler generate [options]

```
Usage: fiddler generate [OPTIONS]

Options:
  -m, --model_path PATH  Directory path for saved model
  --help                 Show this message and exit.
```
  
 
