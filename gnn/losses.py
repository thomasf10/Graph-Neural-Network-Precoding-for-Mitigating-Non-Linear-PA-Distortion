import numpy as np
import tensorflow as tf
from tensorflow import keras
from math import comb


def polynomial_loss(Bs, noisevar, Gw=True):
    """
    :param Bs:  polynomialcoefficients [B1, B3, B5, ..., B2N+1]
    :param noisevar:
    :param Gw: if False use the approximation G(w)= I
    :return:
    """
    def loss(H_batch_train, y_pred):
        """
        :param H_batch_train: batch of training channels (bs x M x K x 2)
        :param y_pred: output of the neuralnet (bs x M x K x 2)
        :return:
        """
        #(2N+1)th order polynomial
        N = Bs.shape[0] - 1#order of the polynomial [B1, B3, B5, ..., B2N+1]

        #reconstruct the channel
        Hre = H_batch_train[:, :, :, 0]
        Him = H_batch_train[:, :, :, 1]
        H = tf.complex(Hre, Him)

        #reconstruct the precoding matrix
        Wre = y_pred[:, :, :, 0]
        Wim = y_pred[:, :, :, 1]
        W = tf.complex(Wre, Wim) #bs x M x K

        #compute the Bussgang gain matrix G(W)
        Cx = W @ tf.transpose(W, conjugate=True, perm=(0, 2, 1))

        #compute H^T W
        Htrans = tf.transpose(H, perm=(0, 2, 1))

        #todo update GW to work with higher order polynomials
        if Gw:
            G = compute_Gw(Bs, Cx)
            HGW = Htrans @ G @ W

            """manual checks"""
            # Gcheck_third_order = tf.cast(tf.eye(tf.shape(W)[-2]), dtype=tf.complex64) \
            #     + tf.cast(2 * Bs[1], dtype=tf.complex64) \
            #     * tf.linalg.diag(tf.linalg.diag_part(Cx)) # I_m + 2 * b3 * I_m *diag(WWh)

            #
            # Gcheck_fifth_order = tf.cast(tf.eye(tf.shape(W)[-2]) * Bs[0], dtype=tf.complex64) \
            #     + 2 * Bs[1] * tf.cast(tf.eye(tf.shape(W)[-2]), dtype=tf.complex64) \
            #     @ tf.linalg.diag(tf.linalg.diag_part(Cx)) \
            #     + 6 * Bs[2] * tf.cast(tf.eye(tf.shape(W)[-2]), dtype=tf.complex64) \
            #     @ (tf.linalg.diag(tf.linalg.diag_part(Cx))**2)
            # #I_m + 2 * b3 * I_m *diag(WWh) + 6 * b5 * Im * diag(WWh)**2
            # print('G from function: ')
            # print(G)
            # print('G check: ')
            # print(Gcheck_fifth_order)
            # print('test this should be all zeros: ')
            # print(G-Gcheck_fifth_order)
            # print('debug')

        else: #simplify G = I
            HGW = Htrans @ W

        #get desired signal
        desiredsignal = tf.math.abs(tf.linalg.diag_part(HGW)) ** 2

        #get user interference: sum_k |HW|^2 - desiredsignal
        userInterference = tf.reduce_sum(tf.math.abs(HGW) ** 2, axis=2) - desiredsignal

        #compute the disotrtion power
        Hconj = tf.math.conj(H)
        Ce = compute_ce(N, Bs, Cx)#tf.cast(2 * tf.math.abs(b3)**2, dtype=tf.complex64) * Cx * tf.cast(tf.math.abs(Cx)**2, dtype=tf.complex64)
        distortion = tf.cast(tf.linalg.diag_part(Htrans @ Ce @ Hconj), dtype=tf.float32)

        '''manual checks'''
        # # Ce_check_thirdorder = tf.cast(2 * tf.math.abs(Bs[1])**2, dtype=tf.complex64) * Cx * tf.cast(tf.math.abs(Cx)**2, dtype=tf.complex64)
        #M = tf.shape(W)[1]
        # Im = tf.eye(M, batch_shape=[tf.shape(Cx)[0]])
        # diagCx = tf.linalg.diag(tf.linalg.diag_part(Cx))
        # L1 = tf.cast((2 / tf.math.sqrt(tf.constant(2.0))), dtype=tf.complex64)* Bs[1] * tf.cast(Im, dtype=tf.complex64) \
        #      + tf.cast((12/tf.math.sqrt(tf.constant(2.0))), dtype=tf.complex64) * Bs[2] * tf.cast(Im, dtype=tf.complex64) * diagCx
        # L2 = tf.cast((6 / tf.math.sqrt(tf.constant(3.0))), dtype=tf.complex64) * Bs[2] * tf.cast(Im, dtype=tf.complex64)
        # Ce_check_fifthorder = L1 @ (Cx * tf.cast(tf.math.abs(Cx)**2, dtype=tf.complex64)) @ tf.transpose(L1, perm=(0, 2, 1), conjugate=True) \
        #                       + L2 @ (Cx * tf.cast(tf.math.abs(Cx)**4, dtype=tf.complex64)) @ tf.transpose(L2, perm=(0, 2, 1), conjugate=True)
        # print('Ce function: ')
        # print(Ce)
        # print('Ce check: ')
        # print(Ce_check_fifthorder)
        # print('ce check this should be zero: ')
        # print(Ce - Ce_check_fifthorder)

        #compute sinr per user
        sinr = desiredsignal / (userInterference + distortion + noisevar)

        #rate per user
        R = tf.math.log(1 + sinr) / tf.math.log(2.0)

        #sumrate
        Rsum = tf.reduce_sum(R, axis=1)

        return -Rsum
    return loss

def compute_ce(N, Bs, Cx):
    Ce = 0
    for n in range(1, N+1):
        Ln = compute_Ln(n, N, Bs, Cx)
        Ce += Ln @ Cx * tf.cast(tf.abs(Cx)**(2*n), dtype=tf.complex64) @ tf.transpose(Ln, perm=(0, 2, 1), conjugate=True)
    return Ce

def compute_Ln(n, N, Bs, Cx):
    """
    :param n:
    :param N:
    :param Bs: array of polynomial coeffs [B1, B3, B5, ..., B2N+1]
    :param M:
    :return:
    """
    M = Cx.shape[1]
    diagCx = tf.linalg.diag(tf.linalg.diag_part(Cx))
    Im = tf.eye(M, batch_shape=[tf.shape(Cx)[0]])

    #compute Ln =
    Ln = 0
    for l in range(n, N+1):
        Ln += tf.cast(comb(l, n) * np.math.factorial(l+1), dtype=tf.complex64) * Bs[l] * \
              tf.cast(Im, dtype=tf.complex64) * diagCx**(l-n)
    Ln *= (1 / np.sqrt(n+1))
    return Ln

def compute_Gw(Bs, Cx):
    """
    :param Bs: poly coefs [b1, b3, b5, ..., b2N+1]
    :param Cx: input covariance matrix
    :return: bussgang gain matrix
    """
    M = Cx.shape[1]
    diagCx = tf.linalg.diag(tf.linalg.diag_part(Cx))
    Im = tf.eye(M, batch_shape=[tf.shape(Cx)[0]])

    #compute G(W) = B1 Im + (n+1)! B3 Im diag(Cx) + ... + (N+1)! B2N+1 diag(Cx)^N
    Gw = tf.cast(np.math.factorial(1), dtype=tf.complex64) * Bs[0] * tf.cast(Im, dtype=tf.complex64) #1e order term
    for n in range(1, Bs.shape[0]):#higher order terms
        Gw += tf.cast(np.math.factorial(n+1), dtype=tf.complex64) * Bs[n] * tf.cast(Im, dtype=tf.complex64) @ diagCx**n
    return Gw