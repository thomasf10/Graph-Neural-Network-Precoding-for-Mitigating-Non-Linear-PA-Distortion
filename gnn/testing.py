import numpy as np
import matplotlib.pyplot as plt
import os
from utils.utils import load_params, C2R, getSymbols, create_folder, get_data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from gnn.losses import polynomial_loss
from gnn.model import GNN_layer, Pwr_norm_gnn
from tqdm import tqdm
from tqdm import trange
import tikzplotlib
import sys


def rapp_amam(Xamp, G, s, psat):
    return G * Xamp / ((1 + np.abs(G * Xamp / np.sqrt(psat)) ** (2*s)) ** (1 / (2*s)))

def rapp_ampm(Xamp, A, B, q):
    """
    :param Xamp: input amplitude
    :param A: fitting param todo: adjust it
    :param B: fitting param todo: adjust it
    :param q: fitting param todo: adjust it
    :return: ampm distortion
    """
    return A * Xamp ** q / (1 + (Xamp / B) ** q)

def rapp_amam_ampm(X, s=2.0, q=4, A=-0.315, B=1.1368585184599176, psat=1, G=1):
    amp = rapp_amam(np.abs(X), G, s, psat)
    phase = rapp_ampm(np.abs(X), A, B, q)
    y = amp * np.exp(1j * (phase + np.angle(X)))
    return y

def avg_sum_rate(W, H, noise_vars, nrdata=1000, Srapp=2.0, plotpwr=False, pa='poly', b3=None, Bs=None, psat=1, backoff=-3):
    """
    :param W: precoding matrix (M x K)
    :param H: channel matrix (M x K)
    :param noise_vars: array of noise variances over which to evaluate the avg sum rate
    :param nrdata: nr of datapoints over which to average the sum rate
    :param plotpwr: plot the power alocation per antenna yes or no
    :param pa: type of PA options: ['poly', 'rapp', 'lin']
    :param b3: third order coeficient
    :param Bs: polynomial coefficients for higher order PA models [b1, b3, ..., b2N+1]
    :return: compute avg sumrate using bussgang at different snr points
    """

    # generate symbols => variance = 1 => E[|wk sk|^2] = Pk
    K = W.shape[-1]
    S = np.zeros((K, nrdata), dtype=complex)
    for k in range(K):
        S[k, :] = getSymbols(nrdata, p=1)
    #pwr_s = np.linalg.norm(S) ** 2 / (K * nrdata)  # check

    # precode
    x = W @ S  # => M x Ndata

    #plot the power allocation per antenna
    if plotpwr:
        if K == 1:
            plt.stem(np.abs(W) ** 2)
            plt.title('nn power per antenna')
            plt.ylabel('|x|^2')
            plt.xlabel('m')
            plt.show()
        else:
            plt.stem(np.sum(np.abs(W) ** 2, axis=1))
            plt.title('nn power per antenna')
            plt.ylabel('|x|^2')
            plt.xlabel('m')
            plt.show()

    # amplify
    if pa == 'poly':
        if b3:
            y = x + b3 * x * np.abs(x) ** 2
        elif Bs is not None: #todo check this amplification for higher order
            #amplify with higher order polynomial
            y = 0
            for n, b in enumerate(Bs):
                # print(f'b: {b}')
                # print(f'n: {n}')
                y += b * x * np.abs(x)**(2*n)
    elif pa == 'rapp': #todo check the rapp model
        psat = 1 / (10 ** (backoff / 10))
        y = rapp_amam_ampm(x, psat=psat, s=Srapp)#rapp_model(x, psat, Srapp)
        print(f'using rapp PA, S:{Srapp} - backoff: {backoff} -psat: {psat}')

    elif pa == 'rapp_amam_only':
        psat = 1 / (10 ** (backoff / 10))
        amp = rapp_amam(np.abs(x), 1, Srapp, psat)
        y = amp * np.exp(1j * (np.angle(x)))

    elif pa == 'lin':
        y = x

    elif pa == 'softlim':
        #backoff = -3#db
        psat = 1 / (10**(backoff/10))
        #psat = 1 #change to change the back-off value
        #print(f'x before softlim: {x}')
        y = np.zeros_like(x, dtype=complex)
        for m in range(x.shape[0]):
            for i in range(x.shape[1]):
                if np.abs(x[m, i]) <= np.sqrt(psat):
                    y[m, i] = x[m, i]
                else:
                    phase = np.angle(x[m, i]) #no ampm distortion
                    #phase = rapp_ampm(np.abs(x[m, i]), -0.315, 1.1368585184599176, 4)#add some ampmdistortion to see effect
                    y[m, i] = np.sqrt(psat) * np.exp(1j * phase) #clip magnitude to 1
        #print(f'softlim: psat: {psat}')

    else:
        raise Exception("SELECT A VALID PA MODEL, OPTIONS ARE: ['poly', 'rapp', 'lin']")

    # send over channel
    r = H.T @ y  # => K x Nrdata

    # linear equalization
    Shat = np.zeros((K, nrdata), dtype=complex)
    sindr = np.zeros((K, len(noise_vars)), dtype=complex)
    for k in range(K):
        Sk = S[k, :]
        rk = r[k, :]
        G = np.mean(Sk * rk.conj()) / np.mean(Sk * Sk.conj())
        Css = np.mean(Sk * Sk.conj()) #should be 1
        sig2_s = np.real(np.abs(G) ** 2 * Css)
        sig2_id = np.real(np.mean(rk * rk.conj()) - np.abs(G) ** 2 * Css)
        sindr[k, :] = sig2_s / (sig2_id + noise_vars)
        # print(f'user {k}, sindr = {10 * np.log10(sindr[k])} dB')
        Shat[k, :] = 1 / np.abs(G) * rk  # r = G s => shat = 1/G * r
        # print(f'S: {Sk}')
        # print(f'Shat: {Shat[k, :]}')

    R = np.real(np.sum(np.log2(1 + sindr), axis=0))
    #print(f'sumrate: {R}')

    return R

def evaluate(sim_params, Htest, model, path=None):
    '''
    :param model: trained model for precoding
    :param Bs: poly coefs: [B1, B2, ..., B2N+1]
    :param H: channels (bs x m x k)
    :param pa: pa type
    :param zf: if true add zero forcing as a benchmark
    :param lin: if true add zf with linear pa as a benchmark
    :return: evaluation of the model on the testset, compared to ZF
    '''

    #unpack sim params
    nr_snr_points = sim_params['nr_snr_points']
    M = sim_params['M']
    K = sim_params['K']
    Pt = sim_params['Pt']
    zf = sim_params['zf']
    lin = sim_params['lin']
    title_add = sim_params['title_add']
    pa = sim_params['pa']
    Bs = sim_params['Bs']
    dpd = sim_params['dpd']
    #todo when using Rapp model provide a way to pass the params of the RAPP model
    if 'srapp' in sim_params.keys() and 'backoff' in sim_params.keys():
        Srapp = sim_params['srapp']
        backoff = sim_params['backoff']
    else:
        Srapp = -1
        backoff = None

    #convert channel to correct shape for the nn
    Htest_real = C2R(Htest)
    Htest_flat = tf.reshape(Htest_real, [Htest.shape[0], -1, 2])

    #set nr of snr points
    snr_tx = np.linspace(-30, 35, nr_snr_points)
    noise_vars = Pt / (10 ** (snr_tx / 10))
    snr_tx_final = 10 * np.log10(Pt / noise_vars)

    #containers to store the sumrates
    Rnn = np.zeros((Htest.shape[0], nr_snr_points))
    if zf:
        Rzf = np.zeros((Htest.shape[0], nr_snr_points))
    if lin:
        Rzflin = np.zeros((Htest.shape[0], nr_snr_points))
    if dpd:
        Rzfdpd = np.zeros((Htest.shape[0], nr_snr_points))
    if K == 1:
        add_zero = True
        Rzero = np.zeros((Htest.shape[0], nr_snr_points))
    else:
        add_zero = False

    print(f'starting evaluation for {pa} PA with params: {Bs}')

    #run the neural net to get precoding vectors for all channel realizations
    y_preds = model.predict(Htest_flat)

    #loop over the channel realizations
    for i in trange(Htest.shape[0], desc='channel realizations'):

        # compute precoding coeff with the nn
        y_pred = y_preds[i, :, :, :] #(1, M, K, 2)
        Wre = y_pred[:, :, 0]
        Wim = y_pred[:, :, 1]
        Wnn = Wre + 1j * Wim  # M x K

        # compute the avg sumrate for the neural network precoder
        Rnn[i, :] = avg_sum_rate(Wnn, Htest[i, :, :], noise_vars, pa=pa, Srapp=Srapp, b3=None, Bs=Bs, backoff=backoff)

        #zero forcing
        if zf:
            Hcomplex = Htest[i, :, :]#np.squeeze(Htest[i, :, :])
            Wzf = Hcomplex.conj() @ np.linalg.inv(Hcomplex.T @ Hcomplex.conj())  # M x K
            norm = np.sqrt(Pt / np.linalg.norm(Wzf, ord='fro') ** 2)
            Wzf *= norm
            Rzf[i, :] = avg_sum_rate(Wzf, Htest[i, :, :], noise_vars, pa=pa, Srapp=Srapp, b3=None, Bs=Bs, backoff=backoff)

        #add zf with linear pa as benchmark
        if lin:
            Hcomplex = Htest[i, :, :]#np.squeeze(Htest[i, :, :])
            Wzf = Hcomplex.conj() @ np.linalg.inv(Hcomplex.T @ Hcomplex.conj())  # M x K
            norm = np.sqrt(Pt / np.linalg.norm(Wzf, ord='fro') ** 2)
            Wzf *= norm
            Rzflin[i, :] = avg_sum_rate(Wzf, Htest[i, :, :], noise_vars, pa='lin', Srapp=Srapp, b3=None, Bs=Bs)

        # add zf with linear pa as benchmark
        if dpd:
            Hcomplex = Htest[i, :, :]  # np.squeeze(Htest[i, :, :])
            Wzf = Hcomplex.conj() @ np.linalg.inv(Hcomplex.T @ Hcomplex.conj())  # M x K
            norm = np.sqrt(Pt / np.linalg.norm(Wzf, ord='fro') ** 2)
            Wzf *= norm
            Rzfdpd[i, :] = avg_sum_rate(Wzf, Htest[i, :, :], noise_vars, pa='softlim', Srapp=Srapp, b3=None, Bs=Bs, backoff=backoff)

        #add z3ro precoder when we only have 1 user
        if add_zero:
            Hcomplex = Htest[i, :, :]
            M_s = 1
            gamma = (np.sum(np.abs(Hcomplex[M_s:]) ** 4) / np.sum(np.abs(Hcomplex[0: M_s]) ** 4)) ** (1 / 3)
            Wzero = Hcomplex.conj()
            Wzero[0:M_s] *= (-gamma)
            norm = np.sqrt(Pt / np.linalg.norm(Wzero, ord='fro') ** 2)
            Wzero *= norm
            Rzero[i, :] = avg_sum_rate(Wzero, Htest[i, :, :], noise_vars, Srapp=Srapp, pa=pa, b3=None, Bs=Bs, backoff=backoff)


    #take avgs
    avg_Rs_nn = np.mean(Rnn, axis=0)
    if zf:
        avg_Rs_zf = np.mean(Rzf, axis=0)
    if lin:
        avg_Rs_lin = np.mean(Rzflin, axis=0)
    if add_zero:
        avg_Rs_z3ro = np.mean(Rzero, axis=0)
    if dpd:
        avg_Rs_dpd = np.mean(Rzfdpd, axis=0)

    #plot and save
    print(f'nn avg sum rate: {avg_Rs_nn}')
    plt.plot(snr_tx_final, avg_Rs_nn, label='nn + poly PA')
    create_folder(os.path.join(path, 'evaluation'))
    np.save(os.path.join(path, 'evaluation', 'avg_Rs_nn.npy'), avg_Rs_nn)
    np.save(os.path.join(path, 'evaluation', 'snr_points.npy'), snr_tx_final)

    if zf:
        plt.plot(snr_tx_final, avg_Rs_zf, label='zf + poly PA')
        print(f'zf avg sum rate: {avg_Rs_zf}')
        np.save(os.path.join(path, 'evaluation', 'avg_Rs_zf.npy'), avg_Rs_zf)

    if lin:
        plt.plot(snr_tx_final, avg_Rs_lin, label='zf + lin PA')
        print(f'zf lin pa avg sum rate: {avg_Rs_lin}')
        np.save(os.path.join(path, 'evaluation', 'avg_Rs_lin.npy'), avg_Rs_lin)


    if add_zero:
        plt.plot(snr_tx_final, avg_Rs_z3ro, label='Z3RO + poly PA')
        print(f'z3ro poly pa avg sum rate: {avg_Rs_z3ro}')
        np.save(os.path.join(path, 'evaluation', 'avg_Rs_z3ro.npy'), avg_Rs_z3ro)

    if dpd:
        plt.plot(snr_tx_final, avg_Rs_dpd, label='ZF + DPD')


    plt.legend()
    plt.plot()
    plt.xlabel('SNR Tx [dB]')
    plt.ylabel('R sum [bits/channel use]')
    plt.title(f'M: {M} - K: {K} {title_add}')
    tikzplotlib.clean_figure()
    tikzplotlib.save(os.path.join(path, 'evaluation', 'R_vs_snr.tex'))
    fig = plt.gcf()
    fig.savefig(os.path.join(path, 'evaluation', 'R_vs_snr.pdf'))
    plt.show()


if __name__ == '__main__':
    #define path of the model to be tested
    model_folder = 'stored_models_11_order_test1'
    scenario = 'M_64_K_1_nvar_0.64'
    setup = 'gnn_mean_aggregation_skips_None_BO_-3_8_layers_128_features_lreluM_64_K_1_03-04-2023--17-51-44'
    path = os.path.join(os.getcwd(), model_folder, scenario, setup)

    path = r'D:\thomas.feys\GNN_precoder_snr_input\gnn\stored_models_11_order_test1\M_64_K_2_nvar_0.64\gnn_mean_aggregation_skips_None_BO_-3_8_layers_128_features_lreluM_64_K_2_03-04-2023--19-19-31'

    #load simulation params
    params = load_params(os.path.join(path, 'mimo_params.json'))
    trainingparams = load_params(os.path.join(path, 'train_params.json'))

    # load model
    model = keras.models.load_model(
        os.path.join(path, 'model.h5'),
        custom_objects={
            'GNN_layer': GNN_layer,
            'Pwr_norm_gnn': Pwr_norm_gnn,
            'loss': polynomial_loss
        }
    )
    print(model.summary())

    #load poly params
    polyparams = np.load(os.path.join(path, 'Bs.npy'))
    print(f'poly params: {polyparams}')

    #get the params out
    M = params['M']
    K = params['K']
    Pt = params['Pt']
    Ntr = trainingparams['Nr_train']
    Nval = trainingparams['Nr_val']
    Nte = trainingparams['Nr_test']

    # load the data
    datafolder = os.path.join(os.getcwd(), 'datasets')
    if 'channelmodel' in params.keys():
        Htrain, Hval, Htest = get_data(M, K, Ntr, Nval, Nte, datafolder, params['channelmodel'])
    else:
        Htrain, Hval, Htest = get_data(M, K, Ntr, Nval, Nte, datafolder, 'iid')



    # # old way (when we stored it per new model)
    # datapath = os.path.join(path, 'data')
    # Htest = np.load(os.path.join(datapath, 'Htrain.npy'))  # todo change to validation set

    #set simulation parameters
    nr_snr_points = 24
    sim_params = {
        'nr_snr_points': 24,
        'Pt': Pt,
        'M': M,
        'K': K,
        'Bs': polyparams,
        'zf': True,
        'lin': True,
        'dpd': False,
        'title_add': 'testset',
        'pa': 'poly',#'softlim', #'rapp', #poly
        'srapp': 2,
        'backoff': -3
    }
    print(f'starting simultaion for: {sim_params}')

    #run evaluation
    #evaluate(Htest[0:250, :, :], model, b3, path=path, Pt=Pt, M=M, K=K) #when using old third order model
    evaluate(sim_params, Htest[0:10, :, :], model, path=os.path.join(path, f"testsetEvaluation{sim_params['pa']}"))

    ''' todo
    - test with lin pa if the system model is oke
    - run with nn
    - make a function to do it with zf and mrt
    - avg over multiple channel realizations
    '''
