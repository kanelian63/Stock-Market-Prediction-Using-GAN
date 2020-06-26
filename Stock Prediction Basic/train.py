
import numpy as np

import utils

from hyperparameters import hp



def main() :

    if hp.size_stocks == 0 :
        
        data = np.load(hp.path_dir + '/' + hp.code +'_from_2010.npy')
        batch_gen = utils.single_stock_generator(data, hp.batch_size, hp.M, hp.N, 
                                                hp.seq_len, hp.n_features, hp.num_stock_size)
    else :
        batch_gen = utils.multi_stock_generator(hp.path_dir, hp.batch_size, hp.M, hp.N,
                                    hp.seq_len, hp.n_features, hp.num_stock_size)

    print("Stock sequence length is {}, and stock size is {}.".format(hp.seq_len, hp.num_stock_size))

    if hp.model_type == "gan" :
        import gan
        if not hp.size_stocks == 1 :
            print("Sorry, gan model work only with multi stocks.")
            import sys; sys.exit()
        gan.gan_train_step(batch_gen, hp.num_epochs, hp.num_stock_size, hp.num_iters, hp.M, hp.N, hp.T)
    else :
        import dnn
        model = dnn.train_step(batch_gen, hp.num_epochs, hp.num_stock_size, hp.num_iters)


    if hp._savehistory :
        num_test_iters = hp.num_iters // 4
        utils.saveHistory(batch_gen, model, 
                            M=hp.M, num_stock_size=hp.num_stock_size,
                            num_test_iters=num_test_iters)

    print("All training step is done.")



if __name__ == "__main__" :

    main()    


