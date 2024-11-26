import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from utils.utils import C2R, logparams, create_folder, logmodel, get_data
from gnn.losses import polynomial_loss
from datetime import datetime
from gnn.model import GNN_layer, Pwr_norm_gnn, Efficient_GNN_layer
from gnn.testing import evaluate
from gnn.naming import get_name
from gnn.activations import get_activation
from gnn.callbacks import Grad_tb_callback, monitor_weights_and_grads
import sys
import matplotlib as mpl



def get_efficient_GNN(M, K, feature_size, Pt, layers, activation='lrelu', aggregation='sum'):
    input_feature_size = 2

    # construct model
    model = keras.Sequential()
    model.add(keras.Input(shape=(M * K, 2)))
    layer_name = get_name('efficient_gnn_layer', nr=0, activation_string=activation)

    # input layer
    model.add(
        Efficient_GNN_layer(input_feature_size, feature_size, M, K, nr=0, act=activation, name=layer_name,
                  aggregation=aggregation))  # add first layer

    # hidden layers
    for l in range(layers - 2):
        layer_name = get_name('efficient_gnn_layer', nr=l + 1, activation_string=activation)
        model.add(Efficient_GNN_layer(feature_size, feature_size, M, K, nr=l + 1, act=activation,
                            name=layer_name, aggregation=aggregation))

    layer_name = get_name('efficient_gnn_layer', nr=layers)

    # output layer
    model.add(Efficient_GNN_layer(feature_size, 2, M, K, nr=layers, name=layer_name, aggregation=aggregation))
    model.add(Pwr_norm_gnn(Pt, M, K))

    return model


def get_GNN(M, K, feature_size, Pt, layers, activation='lrelu', aggregation='sum'):
    input_feature_size = 2

    # construct model
    model = keras.Sequential()
    model.add(keras.Input(shape=(M * K, 2)))
    layer_name = get_name('gnn_layer', nr=0, activation_string=activation)

    # input layer
    model.add(
        GNN_layer(input_feature_size, feature_size, M, K, nr=0, act=activation, name=layer_name,
                  aggregation=aggregation))  # add first layer

    # hidden layers
    for l in range(layers - 2):
        layer_name = get_name('gnn_layer', nr=l + 1, activation_string=activation)
        model.add(GNN_layer(feature_size, feature_size, M, K, nr=l + 1, act=activation,
                            name=layer_name, aggregation=aggregation))

    # output layer
    layer_name = get_name('gnn_layer', nr=layers)
    model.add(GNN_layer(feature_size, 2, M, K, nr=layers, name=layer_name, aggregation=aggregation))
    model.add(Pwr_norm_gnn(Pt, M, K))

    return model


def train(training_params, sim_params, mainfolder='stored_models'):
    """
    :param training_params:
    :param sim_params:
    :return: train for a certain simulation and training set up
    """
    # unpack simulation parameters
    M = sim_params['M']
    K = sim_params['K']
    Pt = sim_params['Pt']
    noise_var = sim_params['noise_var']
    Bs = sim_params['Bs']
    ch_model = sim_params['channelmodel']
    order = 2 * (Bs.shape[0] - 1) + 1
    print(f'order: {order}')

    # unpack training parameters
    layertype = training_params['layer_type']
    Ntr = training_params['Nr_train']
    Nval = training_params['Nr_val']
    Nte = training_params['Nr_test']
    lr = training_params['lr']
    layers = training_params['layers']
    eager_mode = training_params['eager_mode']
    earlystopping = training_params['early_stop']
    reduce_lr = training_params['reduce_lr']
    feature_size = training_params['dl']
    activation = training_params['activation']
    monitor_weights_and_grads_callback = training_params['monitor_weights_and_grads']
    tensorboard_gradmon = training_params['tensorboard']
    aggregation = training_params['aggregation']

    # create results folder
    subfolder = f"M_{M}_K_{K}_nvar_{noise_var}"
    output_dir = os.path.join(
        os.getcwd(),
        mainfolder,
        subfolder,
        f"{training_params['note']}M_{M}_K_{K}_{datetime.now().strftime('%d-%m-%Y--%H-%M-%S')}"
    )

    # store the simulation params
    create_folder(output_dir)
    logparams(os.path.join(output_dir, 'mimo_params.json'), sim_params, output_dir=output_dir)
    logparams(os.path.join(output_dir, 'train_params.json'), training_params)

    # generate channels
    datafolder = os.path.join(os.getcwd(), 'datasets')
    Htrain, Hval, Htest = get_data(M, K, Ntr, Nval, Nte, datafolder, channelmodel=ch_model)

    # convert complex numbers to real numbers bs x M x K => bs x M x K x 2
    Htrainre = C2R(Htrain)
    Htestre = C2R(Htest)
    Hvalre = C2R(Hval)

    # todo: dont need this for CCNN
    # flatten for GNN (not needed for CCNN) bs x M x K x 2 => bs x MK x 2
    Htrain_flat = tf.reshape(Htrainre, [Ntr, -1, 2])
    Hval_flat = tf.reshape(Hvalre, [Nval, -1, 2])

    # make tf datasets
    # #todo: note that the input to the nn is flat while the label aka input to the lossfunction ground truth is not flat! check what is needed when using CCNN
    train_dataset = tf.data.Dataset.from_tensor_slices((Htrain_flat, Htrainre))
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((Hval_flat, Hvalre))
    val_dataset = val_dataset.batch(batch_size)

    # define the GNN model
    if layertype == 'gnn':
        model = get_GNN(M, K, feature_size, Pt, layers, activation,
                        aggregation=aggregation)
        print(model.summary())
    elif layertype == 'efficient_gnn':
        model = get_efficient_GNN(M, K, feature_size, Pt, layers, activation,
                        aggregation=aggregation)
        print(model.summary())
    else:
        assert False, f'invalid layer type: {layertype}'

    # select optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # compile the model
    loss = polynomial_loss(Bs, noise_var, Gw=True)
    model.compile(loss=loss, optimizer=optimizer, run_eagerly=eager_mode)
    print(f'model loss function: {model.loss.__name__}')

    # add callbacks todo check if we can use this, see unfolding model
    save = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, "model.h5"),
        monitor='val_loss',
        verbose=0,
        save_best_only=True
    )

    log = tf.keras.callbacks.CSVLogger(os.path.join(output_dir, 'training.log'))

    callbacks = [save, log]

    if earlystopping:
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        callbacks.append(earlystop)
    if reduce_lr:
        reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_delta=0.001,
                                                        verbose=1)
        callbacks.append(reducelr)
    if tensorboard_gradmon:
        # create logs folder
        create_folder(os.path.join(output_dir, 'tb_logs'))
        # Define the TensorBoard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(os.path.join(output_dir, 'tb_logs'))
        # Define the custom gradients callback
        gradients_callback = Grad_tb_callback(log_dir=os.path.join(output_dir, 'tb_logs'), histogram_freq=1,
                                              val_dataset=val_dataset)
        callbacks.append(tensorboard_callback)
        callbacks.append(gradients_callback)

    # log the model summary
    logmodel(os.path.join(output_dir, 'model.json'), model)

    print(f'*******************************************START TRAINING FOR:*******************************************')
    print(f'{sim_params} - {training_params}')
    history = model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )

    # plot loss progress
    plt.plot(-np.array(history.history['loss']))
    plt.plot(-np.array(history.history['val_loss']))
    plt.title(f'sum rate M: {M} K: {K}')
    plt.ylabel('R')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig = plt.gcf()
    fig.savefig(os.path.join(output_dir, 'model_loss.pdf'))
    plt.show()

    print(f'training done for M:{M}, K:{K}')
    print(f'evaluating on Hval for poly coeffs: {Bs}')

    if layertype == 'gnn':
        # load model (to get best weights)
        bestModel = keras.models.load_model(
            os.path.join(output_dir, 'model.h5'),
            custom_objects={
                'GNN_layer': GNN_layer,
                'Pwr_norm_gnn': Pwr_norm_gnn,
                'loss': polynomial_loss
            }
        )
    elif layertype == 'efficient_gnn':
        # load model
        bestModel = keras.models.load_model(
            os.path.join(output_dir, 'model.h5'),
            custom_objects={
                'Efficient_GNN_layer': Efficient_GNN_layer,
                'Pwr_norm_gnn': Pwr_norm_gnn,
                'loss': polynomial_loss
            }
        )

    print(bestModel.summary())

    # evaluate the neural nets performance
    eval_params = {
        'nr_snr_points': 24,
        'Pt': Pt,
        'M': M,
        'K': K,
        'Bs': Bs,
        'zf': True,
        'lin': True,
        'title_add': 'val set after training',
        'pa': 'poly',
        'dpd': False
    }
    evaluate(eval_params, Hval, bestModel, path=output_dir)

    print(f'done evaluating on Hval for poly coeffs: {Bs}')
    print(f'*******************************************DONE TRAINING FOR:*******************************************')
    print(f'{sim_params} - {training_params}')


if __name__ == '__main__':
    # simulation setup
    channel_model = 'iid' #'los'
    M = 64  # nr tx antennas
    K = 2  # nr users
    Pt = M  # total power
    snr_tx = 20  # in db
    noise_var = Pt / (10 ** (snr_tx / 10))

    # training params
    layer_type = 'gnn'
    Ntr = 1000 #500000 # nr training data
    Nval = 200 #2000
    Nte = 200 #10000 # nr test data
    nr_layers = 8
    hidden_features = 128
    batch_size = 64
    epochs = 2  # 50
    lr = 5e-3
    activation = 'lrelu'
    aggregation_string = 'mean' #'sum'
    note = f'GNN'

    # training settings
    eager_mode = False
    earlystopping = False
    reduce_lr = True
    monitor_weight_and_grad_manually = False
    tensorboard = False # does not work with onedrive folders!

    # PA parameters
    '''In case of third order model'''
    # # pa params for -9 , -7.5, -6, -4.5, -3, -1.5, 0 db backof for a third order model
    # backoffs = [-9, -7.5, -6, -4.5, -3, -1.5, 0]
    # different_backoffs = [np.array([1, -0.01993004 - 0.01079656 * 1j]), np.array([1, -0.03068619 - 0.01884522 * 1j]),
    #                       np.array([1, -0.04226415 - 0.02490772 * 1j]), np.array([1, -0.05612979 - 0.03005297 * 1j]),
    #                       np.array([1, -0.07781605 - 0.0401193 * 1j]), np.array([1, -0.11728841 - 0.0650147 * 1j]),
    #                       np.array([1, -0.15905536 - 0.07924839 * 1j])
    #                       ]


    # 11th order parameters for different backoffs  for -9 , -7.5, -6, -4.5, -3, -1.5, 0 db backof
    pa_params_diff_ibos = [
        np.array([1, -4.38184836e-02 - 1.01466832e-01 *1j  , 1.50490437e-03 + 8.42208488e-03 *1j,
         - 3.13452827e-05 - 2.81868627e-04*1j,  3.49967293e-07 + 4.20633310e-06*1j,
         - 1.59432984e-09 - 2.31868139e-08*1j]) ,
        np.array([1, -5.79334438e-02-9.36769411e-02*1j,  2.39315994e-03+7.94859107e-03*1j,
         -5.57663136e-05-2.69264129e-04*1j,  6.65066314e-07+4.04837957e-06*1j,
         -3.14808144e-09-2.24280442e-08*1j]),
        np.array([1, -7.50994886e-02-8.42352484e-02*1j,  3.66782506e-03+7.26453523e-03*1j,
         -9.54049052e-05-2.48371067e-04*1j,  1.22703316e-06+3.75613932e-06*1j,
         -6.13183499e-09-2.08924283e-08*1j]),
        np.array([1, -9.35828409e-02-7.41305601e-02*1j,  5.16172165e-03+6.46522185e-03*1j,
         -1.44481282e-04-2.22483069e-04*1j,  1.94963213e-06+3.37874265e-06*1j,
         -1.00752209e-08-1.88479147e-08*1j]),
        np.array([1, -1.11143930e-01 - 6.30816977e-02 * 1j, 6.60156653e-03 + 5.47141526e-03 * 1j,
          -1.91451680e-04 - 1.86610370e-04 * 1j, 2.62822435e-06 + 2.80380833e-06 * 1j,
          -1.36811147e-08 - 1.54579691e-08 * 1j]),
        np.array([1, -1.29033190e-01-5.49758824e-02*1j,  8.21176444e-03+4.85204392e-03*1j,
         -2.48588087e-04-1.68144990e-04*1j,  3.52215545e-06+2.56527492e-06*1j,
         -1.88139985e-08-1.43562319e-08*1j]),
        np.array([1, -1.44473655e-01-4.67375592e-02*1j,  9.58442261e-03+4.13617338e-03*1j,
         -2.96362436e-04-1.43570171e-04*1j,  4.25309097e-06+2.19271142e-06*1j,
         -2.29128062e-08-1.22805850e-08*1j])
    ]
    ibos = [-9, -7.5, -6, -4.5, -3, -1.5, 0]

    # create mapping from ibo to pa params
    ibo_to_Bs = {ibo:poly_param for ibo, poly_param in zip(ibos, pa_params_diff_ibos)}

    # select ibo
    used_ibo = -3
    Bs = ibo_to_Bs[used_ibo]

    # put all the params in a dictionary to store it
    sim_params = {
        'M': M,
        'K': K,
        'Pt': Pt,
        'snr_tx': snr_tx,
        'noise_var': noise_var,
        'Bs': Bs,
        'ibo': used_ibo,
        'channelmodel': channel_model
    }

    training_params = {
        'layer_type': layer_type,
        'Nr_train': Ntr,
        'Nr_val': Nval,
        'Nr_test': Nte,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'reduce_lr': reduce_lr,
        'early_stop': earlystopping,
        'eager_mode': eager_mode,
        'monitor_weights_and_grads': monitor_weight_and_grad_manually,
        'tensorboard': tensorboard,
        'layers': nr_layers,
        'dl': hidden_features,
        'activation': activation,
        'aggregation': aggregation_string,
        'note': note
    }

    # snr values at witch we train P_t/noisevar
    snr_tx_set = np.array([20])  # np.array([-10, 0, 10, 20, 30]) #in db
    noise_var_set = Pt / (10 ** (snr_tx_set / 10))
    print(f'noisevarset: {noise_var_set}')

    # folder to store results
    folder_for_test = f'stored_models_{channel_model}_channel'
    aggregation_string = 'mean'

    # start training for different configurations

    print(f"Pt: {sim_params['Pt']}")
    print(f'training at an IBO of {used_ibo} with PA: {Bs}')

    #set seed the same initializations when comparing eg activations, etc
    tf.random.set_seed(42)

    # start training
    train(training_params, sim_params, mainfolder=folder_for_test)
