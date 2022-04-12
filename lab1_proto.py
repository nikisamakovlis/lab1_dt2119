# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.signal import lfilter
from scipy.signal.windows import hamming
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from scipy.spatial.distance import euclidean
from sklearn.mixture import GaussianMixture
from lab1_tools import *

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    # take the samples, copy them in columns in the matrix, aranged into columns, for every window we want to consider have a column in this matrix
    # and then we can apply the subsequence processing to each column of this representation

    signal_length = len(samples)

    # create numpy array [N x winlen], where N is the number of windows that fit in the input signal
    
    N = int(np.ceil((signal_length - winlen + winshift)/winshift ))
    total_rows = N - 1
    i = 0
    frame_matrix = np.zeros((total_rows, winlen))
    for row in range(0, total_rows):   # Row indeces go from 0 to 91 (tot 92)
        for col in range(0, winlen):   # Col indeces go from 0 to 399 (tot 400)

            frame_matrix[row][col] = samples[i]
            i += 1
            if i == signal_length - winlen: # last window reached
                break
        i -= winshift
    return frame_matrix
    #print(frame_matrix.shape)
    #plt.pcolormesh(frame_matrix)
    #plt.show()
    
def preemp(matrix, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """

    preemp_matrix = np.zeros(matrix.shape)

    a = [1]
    b = [1, -p]
    for i in range(len(matrix)):
        preemp_matrix[i] = lfilter(b, a, matrix[i])

    #plt.pcolormesh(preemp_matrix)
    #plt.show()
    return preemp_matrix

def windowing(matrix):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windowed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    
    # define hamming window 
    window = hamming(len(matrix[0]), sym=False)
    
    windowed_matrix = np.zeros(matrix.shape)

    for i in range(len(matrix)):
        windowed_matrix[i] = matrix[i] * window

    #plt.pcolormesh(matrix)
    #plt.show()

    #plt.pcolormesh(windowed_matrix)
    #plt.show()

    return windowed_matrix


def powerSpectrum(matrix, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    fft_matrix = np.zeros((len(matrix), nfft))

    for i in range(len(matrix)):
        fft_matrix[i] = abs(fft(matrix[i], n = nfft))**2
    
    #plt.pcolormesh(fft_matrix)
    #plt.show()
    return fft_matrix
    
    

def logMelSpectrum(matrix, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    nfft = len(matrix[0])
    fbank = trfbank(samplingrate, nfft)

    # plot the 40 filters of in the bank
    #for i in range(len(fbank)):
    #    plt.plot(fbank[i])
    #plt.show()

    mel_matrix = np.log(matrix @ fbank.T)
    #plt.pcolormesh(mel_matrix)
    #plt.show()

    return mel_matrix


def cepstrum(matrix, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """

    dct_matrix = np.zeros((len(matrix), nceps))

    for i in range(len(matrix)):
        dct_temp = dct(matrix[i])
        dct_matrix[i] = dct_temp[:nceps] # selecting the 13 first coefficients from the dct results
    
    ceps_matrix = lifter(dct_matrix)
    
    return ceps_matrix
    #plt.pcolormesh(ceps_matrix)
    #plt.show()

def euclidian_distances(x, y):
    '''
    input: x,y are utterences (not gone through mfcc)
    ouptut: matrix with euclidian distances 

    '''

    x_coefficient_matrix = mfcc(x)
    N = len(x_coefficient_matrix)

    y_coefficient_matrix = mfcc(y)
    M = len(y_coefficient_matrix)

    local_euclidean_matrix = np.zeros((N,M))

    for n in range(N):
        for m in range(M):
            local_euclidean_matrix[n, m] = euclidean(x_coefficient_matrix[n, :], y_coefficient_matrix[m, :])

    return local_euclidean_matrix



def dtw(local_euclidean_matrix, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    N = len(local_euclidean_matrix)
    M = len(local_euclidean_matrix[0])
    AccD = np.zeros((N, M))
    for n in range(N):
        for m in range(M):
            AccD[n, m] = local_euclidean_matrix[n, m] + min(AccD[n-1, m], AccD[n-1, m-1], AccD[n, m-1])
'''
def main():
    example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()
    samplingRate = example['samplingrate']

    # Desired window length in time: 20 ms = 20 * 10^-3 s
    # Desired shift length in time: 10 ms =  10 * 10^-3 s
    # Sampling rate = 20 000 samples/s
    # 20 milliseconds in samples:  20 * 10^-3 s * 20 000 samples/s = 400 samples
    # 10 milliseconds in samples:  10 * 10^-3 s * 20 000 samples/s = 200 samples

    enframed = enframe(example['samples'], 400, 200)  # example['samples'].shape = (18432,)
    preemped = preemp(enframed)

    windowed = windowing(preemped)

    transformed = powerSpectrum(windowed, 512)

    mel_speced = logMelSpectrum(transformed, samplingRate)

    cepstrum(mel_speced, 13)

    plt.pcolormesh(example['lmfcc'])
    plt.show()

'''

def up_until_mel(utterance, mel_concatenation):

    enframed = enframe(utterance, 400, 200) 

    preemped = preemp(enframed)

    windowed = windowing(preemped)

    transformed = powerSpectrum(windowed, 512)

    mel_speced = logMelSpectrum(transformed, 20000)

    mel_concatenation = np.vstack((mel_concatenation, mel_speced))

    return mel_concatenation

def mfcc(utterance):

    enframed = enframe(utterance, 400, 200) 

    preemped = preemp(enframed)

    windowed = windowing(preemped)

    transformed = powerSpectrum(windowed, 512)

    mel_speced = logMelSpectrum(transformed, 20000)

    cepstrum_coefficient_matrix = cepstrum(mel_speced, 13) # output is a N x 13

    return cepstrum_coefficient_matrix

def gmm_cluster(feature_matrix, data):

    
    component_sizes = [4, 8, 16, 32]
    #component_sizes = [4]

    
    for component_size in component_sizes:
        gm = GaussianMixture(n_components = component_size).fit(feature_matrix)
        
        # 16    ->  "seven" MAN     A
        # 17    ->  "seven" MAN     B
        # 38    ->  "seven" WOMAN   A
        # 39    ->  "seven" WOMAN   B
        seven_utterances = data[[16, 17, 38, 39]]
        
        class_labels = []
        posterior_list = []

        fig, axs = plt.subplots(2, 2)
        

        for utterance in seven_utterances:
            test_data = mfcc(utterance['samples'])
            posterior = gm.predict_proba(test_data)
            posterior_list.append(posterior)

            #predicted_label = gm.predict(test_data)
            #print(predicted_label)
            #print(predicted_label.shape)
            #class_labels.append(predicted_label)
            
            
            #plt.plot(posterior)
            #gender = utterance["gender"]
            #repetition = utterance["repetition"]
            #plt.title(f'Posterior with {component_size} components for #utterance {gender} and repetition {repetition}')
            #plt.show()
        
        '''
        axs[0, 0].pcolormesh(posterior_list[0])
        axs[0, 1].pcolormesh(posterior_list[1])
        axs[1, 0].pcolormesh(posterior_list[2])
        axs[1, 1].pcolormesh(posterior_list[3])

        plt.show()
        
        plt.plot(class_labels[0], label="man")
        plt.plot(class_labels[2], label="woman")
        plt.legend()
        plt.show()
        '''

        



        

            
        

def main():
    data = np.load('lab1_data.npz', allow_pickle=True)['data']
    example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()
 
    concatenation = np.zeros((1,13))
    mel_concatenation = np.zeros((1,40))
    for utterance in data:
        
        mfcc_temp = mfcc(utterance['samples'])
    
        concatenation = np.vstack((concatenation, mfcc_temp))
        mel_concatenation = up_until_mel(utterance['samples'], mel_concatenation)

    
    big_feature_array = np.delete(concatenation, (0), axis=0)   # remove first row of zeros

    #pd_big_feature_array = pd.DataFrame(big_feature_array)
    #correlation_matrix = pd_big_feature_array.corr()

    big_mel_array = np.delete(mel_concatenation, (0), axis=0)
    #pd_big_mel_array = pd.DataFrame(big_mel_array)
    #mel_correlation_matrix = pd_big_mel_array.corr()



    #gmm_cluster(big_feature_array, data)    

    local_euclidian = euclidian_distances(data[16]["samples"], data[17]["samples"])

    dtw(local_euclidian)
    
   

    #print(correlation_matrix)

    #fig, ax = plt.subplots()
    #heatmap = ax.pcolor(correlation_matrix, cmap=plt.cm.Blues)
    #plt.show()

    #fig, ax = plt.subplots()
    #heatmap = ax.pcolor(mel_correlation_matrix, cmap=plt.cm.Blues)
    #plt.show()

    #plt.pcolormesh(correlation_matrix)
    #plt.show()

    #sns.heatmap(correlation_matrix, annot=False, cmap=plt.cm.Reds)
    #plt.show()

    #sns.heatmap(mel_correlation_matrix, annot=False, cmap=plt.cm.Reds)
    #plt.show()

    #plt.pcolormesh(big_feature_array)
    #plt.show()

    #plt.pcolormesh(example['mspec'])
    #plt.show()


        
        

main()