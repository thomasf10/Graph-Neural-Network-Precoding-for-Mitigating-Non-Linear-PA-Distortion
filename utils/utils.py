import os
import numpy as np
import json
import re

def get_data(M, K, Ntr, Nval, Nte, datafolder, channelmodel='iid'):

    create_folder(datafolder)

    #get all dataset names
    all_datasets = [os.path.basename(f.path) for f in os.scandir(datafolder) if f.is_dir()]

    #loop over all existing datasets
    dataexists = False
    for dataset_name in all_datasets:
        #extract dataset information from foldername
        Mdataset = int(re.findall(str(re.escape('M_')) +"(.*)" + str(re.escape('_K')) , dataset_name)[0])
        Kdataset = int(re.findall(str(re.escape('K_')) +"(.*)" + str(re.escape('_Ntr')) , dataset_name)[0])
        Ntr_dataset = int(re.findall(str(re.escape('Ntr_')) +"(.*)" + str(re.escape('_Nval')) , dataset_name)[0])
        Nval_dataset = int(re.findall(str(re.escape('Nval_')) +"(.*)" + str(re.escape('_Nte')) , dataset_name)[0])
        Nte_dataset = int(re.findall(str(re.escape('Nte_')) +"(.*)" + str(re.escape('_')) , dataset_name)[0])

        #check if the desired dataset exists
        if M == Mdataset and K == Kdataset and Ntr <= Ntr_dataset and Nval <= Nval_dataset and Nte <= Nte_dataset and channelmodel in dataset_name:
            Htrain = np.load(os.path.join(datafolder, dataset_name, 'Htrain.npy'))
            Htest = np.load(os.path.join(datafolder, dataset_name, 'Htest.npy'))
            Hval = np.load(os.path.join(datafolder, dataset_name, 'Hval.npy'))
            #only take the data you need
            Htrain = Htrain[0:Ntr, :, :]
            Htest = Htest[0:Nte, :, :]
            Hval = Hval[0:Nval, :, :]
            dataexists = True
            print('Data has been loaded')
            break #exit the loop if we found the correct dataset

    #generate the dataset if it doesn't exist yet
    if dataexists == False:
        if channelmodel == 'iid':
            H = np.zeros((Ntr + Nval + Nte, M, K), dtype=complex)
            for i in range(Ntr + Nte + Nval):
                H[i, :, :] = rayleigh_channel_MU(M, K)

            # generate test and train set
            Htrain = H[0:Ntr, :, :]
            Hval = H[Ntr:Nval + Ntr, :, :]
            Htest = H[Ntr + Nval:Ntr + Nval + Nte, :, :]

            #save the data
            output_dir = os.path.join(datafolder, f'M_{M}_K_{K}_Ntr_{Ntr}_Nval_{Nval}_Nte_{Nte}_iid')
            create_folder(output_dir)
            np.save(os.path.join(output_dir, 'Htrain.npy'), Htrain)
            np.save(os.path.join(output_dir, 'Hval.npy'), Hval)
            np.save(os.path.join(output_dir, 'Htest.npy'), Htest)
            print('Data has been generated')

        elif channelmodel == 'los':
            H = np.zeros((Ntr + Nval + Nte, M, K), dtype=complex)
            for i in range(Ntr + Nte + Nval):
                H[i, :, :] = los_channel_MU(M, K)

            # generate test and train set
            Htrain = H[0:Ntr, :, :]
            Hval = H[Ntr:Nval + Ntr, :, :]
            Htest = H[Ntr + Nval:Ntr + Nval + Nte, :, :]

            #save the data
            output_dir = os.path.join(datafolder, f'M_{M}_K_{K}_Ntr_{Ntr}_Nval_{Nval}_Nte_{Nte}_los')
            create_folder(output_dir)
            np.save(os.path.join(output_dir, 'Htrain.npy'), Htrain)
            np.save(os.path.join(output_dir, 'Hval.npy'), Hval)
            np.save(os.path.join(output_dir, 'Htest.npy'), Htest)
            print('Data has been generated')

        else:
            print(f'{channelmodel} channel model not implemented')


    return Htrain, Hval, Htest


def get_data_snr(M, K, snr_range, Ntr, Nval, Nte, datafolder):

    #get all dataset names
    all_datasets = [os.path.basename(f.path) for f in os.scandir(datafolder) if f.is_dir()]

    #loop over all existing datasets
    dataexists = False
    for dataset_name in all_datasets:
        #extract dataset information from foldername
        Mdataset = int(re.findall(str(re.escape('M_')) +"(.*)" + str(re.escape('_K')) , dataset_name)[0])
        Kdataset = int(re.findall(str(re.escape('K_')) +"(.*)" + str(re.escape('_Ntr')) , dataset_name)[0])
        Ntr_dataset = int(re.findall(str(re.escape('Ntr_')) +"(.*)" + str(re.escape('_Nval')) , dataset_name)[0])
        Nval_dataset = int(re.findall(str(re.escape('Nval_')) +"(.*)" + str(re.escape('_Nte')) , dataset_name)[0])
        Nte_dataset = int(re.search('Nte_(.*)', dataset_name).group(1))
        pattern = r'\[[^]]*\]'
        list_match = re.search(pattern, dataset_name)
        if list_match:
            snr_range_dataset = eval(list_match.group())
        else:
            snr_range_dataset = 'nothinghere'
            print("No list found in string.")

        #check if the desired dataset exists
        if M == Mdataset and K == Kdataset and Ntr <= Ntr_dataset and Nval <= Nval_dataset and Nte <= Nte_dataset and snr_range_dataset == snr_range:
            Htrain = np.load(os.path.join(datafolder, dataset_name, 'Htrain.npy'))
            Htest = np.load(os.path.join(datafolder, dataset_name, 'Htest.npy'))
            Hval = np.load(os.path.join(datafolder, dataset_name, 'Hval.npy'))
            #only take the data you need
            Htrain = Htrain[0:Ntr, :, :, :]
            Htest = Htest[0:Nte, :, :, :]
            Hval = Hval[0:Nval, :, :, :]
            dataexists = True
            print(f'Data has been loaded from: {datafolder} - {dataset_name}')

            break #exit the loop if we found the correct dataset

    #generate the dataset if it doesn't exist yet
    if dataexists == False:
        # SNR points
        snr_points = np.arange(snr_range[0], snr_range[1]+0.1, snr_range[2])
        print(f'snr range of dataset (start, stop, step): {snr_range}')
        print(f'{snr_points=}')
        snr_normalized = snr_points / snr_range[1] #normalize between -1 and 1
        print(f'{snr_points=}')
        #noise_var_points = M / (10 ** (snr_points / 10)) #M=Pt

        H = np.zeros((Ntr + Nval + Nte, M, K, 2), dtype=complex)
        for i in range(Ntr + Nte + Nval):
            # sample a channel
            H[i, :, :, 0] = rayleigh_channel_MU(M, K)

            # sample SNR points
            H[i, :, :, 1] = np.random.choice(snr_normalized)

        # generate test and train set
        Htrain = H[0:Ntr, :, :, :]
        Hval = H[Ntr:Nval + Ntr, :, :, :]
        Htest = H[Ntr + Nval:Ntr + Nval + Nte, :, :, :]

        #save the data
        output_dir = os.path.join(datafolder, f'H_snr_{snr_range}_M_{M}_K_{K}_Ntr_{Ntr}_Nval_{Nval}_Nte_{Nte}')
        create_folder(output_dir)
        np.save(os.path.join(output_dir, 'Htrain.npy'), Htrain)
        np.save(os.path.join(output_dir, 'Hval.npy'), Hval)
        np.save(os.path.join(output_dir, 'Htest.npy'), Htest)
        print('Data has been generated')


    return Htrain, Hval, Htest

def C2R(HC):
    """
    convert complex channel matrix into the re and im parts
    :param H: K x M complex channel matrix
    :return: H: K x M x 2 channel matrix with a real and an imaginary 'color channel'
    """
    Hre = HC.real
    Him = HC.imag
    HR = np.stack((Hre, Him), axis=-1)
    return HR

def create_folder(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")

def losChannel(theta=60*np.pi/100, M=64):
    """
    :param theta: user angle
    :param M: number of antennas
    :return: the channel matrix for a single user LoS channel
    """
    #generate los channel
    h = np.exp(-1j * np.pi * np.cos(theta) * np.arange(M))

    #normalize the channel
    h = h / np.sqrt(np.sum(abs(h) ** 2) / M)

    return h

def rayleigh_channel_MU(M, K):
    """
    :param M: number of antennas
    :param K: number of users
    :return: H (MxK) complex gaussian distributed channel: ~CN(0,1) = ~ N(0,1/2) + N(0,1/2) * j
    """
    variance = 1/2
    stdev = np.sqrt(variance)
    H = np.zeros((M, K), dtype=complex)

    for k in range(K):
        H[:, k] = np.random.normal(0, stdev, M) + 1j * np.random.normal(0, stdev, M)

    return H

def los_channel_MU(M, K):
    H = np.zeros((M, K), dtype=complex)

    for k in range(K):
        userangle = np.random.randint(0, 180)
        theta = userangle * np.pi / 180
        H[:, k] = 1 * np.exp(-1j * np.pi * np.cos(theta) * np.arange(M))

    return H

def getSymbols(Ndata=5000, p = 1):
    """
    :param Ndata: number of symbols to generate
    :param p: signal variance
    :return: Ndata symbols sampled from a complex gaussian with variance p
    Note that this generates a variable which is drawn from a complex gaussian distribution with variance p
    which is equivalent to a + bj with a and b sampled from a gaussian distriobution with variance p/2.
    Here we first sample a and b from a gaussian with mean 0 and variance 1, by multiplying with sqrt(p)/sqrt(2)
    we obtain variance p/2 for both a and b, given that var(constant * X) = constant^2 var(X)
    """
    s = np.sqrt(p) / np.sqrt(2) * (np.random.randn(Ndata) + 1j* np.random.randn(Ndata))
    return s

class NumpyEncoder(json.JSONEncoder):
    #json incoder fur numpy arrays
    #to undo  np.asarray(json_load["a"])
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def logparams(path, params, output_dir=None):
    copyparams = dict(params)
    if 'Bs' in params.keys():
        np.save(os.path.join(output_dir, 'Bs.npy'), copyparams['Bs'])
        del copyparams['Bs']
    f = open(path, "w")
    f.write(json.dumps(copyparams, cls=NumpyEncoder))
    f.close()

def logmodel(path, model):
    summary = str(model.to_json())
    f = open(path, 'w')
    f.write(json.dumps(summary))
    f.close

def logresults(path, nmse_mrt, nmse_neural):
    results = {
        'nmse_mrt': nmse_mrt,
        'nmse_neural': nmse_neural,
        "unit": 'dB'
    }
    f = open(path, "w")
    f.write(json.dumps(results))
    f.close

def nmse(s, shat):
    """
    :param s: transmit symbols
    :param shat: equalized receive symbols
    :return: nmse in dB
    """
    errors = s - shat
    mse = np.mean(np.abs(errors)**2)
    nmse = mse / np.mean(np.abs(s)**2)
    nmsedb = 10*np.log10(nmse)
    return nmsedb

def load_params(path):
    f = open(path)
    params = json.load(f)
    return params
