class hp :
    batch_size = 32
    lr = 0.005
    path_dir = "./pre_data"
    M = 240
    N = 60
    T = 30

    num_epochs = 12
    num_iters = 150 // batch_size
    seq_len = 2200
    n_features = 22
    num_stock_size = (seq_len - M) // N

    size_stocks = 0  #0 = single, #1 = multi
    code = '000660' 


    model_type = "dnn" ##gan, dnn, cnn, rnn, cond_dnn


    b_loss_plot = False
    _task = "class" #class or reg
    _use_indicator = True
    _savehistory = True #Only work with classificaion task

    _gpu = False
