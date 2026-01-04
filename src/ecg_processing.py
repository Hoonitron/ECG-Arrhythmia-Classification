import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs=360, lowcut=0.5, highcut=40):
    b, a = butter(3, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

def extract_beats(signal, r_peaks, pre=100, post=200):
    beats = []
    for r in r_peaks:
        if r-pre > 0 and r+post < len(signal):
            beats.append(signal[r-pre:r+post])
    return np.array(beats)

def extract_features(beats):
    features = []
    for beat in beats:
        mean = np.mean(beat)
        var = np.var(beat)
        rms = np.sqrt(np.mean(beat**2))
        p2p = np.max(beat) - np.min(beat)
        fft = np.abs(np.fft.fft(beat))[:10]
        features.append([mean, var, rms, p2p] + list(fft))
    return np.array(features)
