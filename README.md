# fiddler
An AI Irish music composer


## Install

In order to install `fiddler` run the following command:

    python setup.py install
    
 This will install `fiddler` as a command line tool.
 
 ## Training Recurrent Neural Network
 
    fiddler train_rnn -f <data_path> -b <batch_size> -n <num_steps> -l <lstm_size> -e <train_epochs>
 
 If `fiddler` is not installed as command-line tool, you can use the same command using `python src/cli.py` with same arguments.
 
