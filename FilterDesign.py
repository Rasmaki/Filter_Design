import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import convolve
import matplotlib.pyplot as plt

f, data = read("StarWars3.wav")
data_len = len(data)
Fs = f

cutoff_freq_normalized = 0.12
cutoff_freq = cutoff_freq_normalized * (Fs / 2)
filter_length = 5
print("Calculating...")


def calc_coefficients(vg, n):
    bn = []
    num = (n-1)/2
    for i in range(int(num)+1):
        if i != 0:
            b = 2 * vg * (np.sin(2 * np.pi * i * vg)/(2 * np.pi * i * vg))
            bn.insert(0, b)
        else:
            b = 2 * vg
        bn.append(b)
    return bn


def plot_filter_types(filter_type, raw_data):
    filtered_data = convolve(raw_data, filter_type)
    return filtered_data


filter_tp_a = np.array([1, 2, 1]) / 4
filter_tp_b = np.array([1, 3, 3, 1]) / 8
filter_tp_c = np.array([1, 4, 6, 4, 1]) / 16
filter_hp_a = np.array([1, -2, 1]) / 4
filter_hp_b = np.array([1, -3, 3, -1]) / 8
filter_hp_c = np.array([1, -4, 6, -4, 1]) / 16
filter_tp_d = np.asarray(calc_coefficients(cutoff_freq_normalized, filter_length))

filter_list = [filter_tp_a, filter_tp_b, filter_tp_c, filter_hp_a, filter_hp_b, filter_hp_c, filter_tp_d]

f_data = []
plt.figure()
plt.plot(abs(np.fft.fft(data)))
plt.yscale('log')
plt.xscale('log')
plt.xlim([20, 20000])
plt.ylim([10 ** -10, 10 ** 10])
plt.title("Spectrum - Raw Data")
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')

for j in range(len(filter_list)):
    f_data.append(plot_filter_types(filter_list[j], data))
    plt.figure()
    plt.plot(abs(np.fft.fft(f_data[j])))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([20, 20000])
    plt.ylim([10**-10, 10**10])
    plt.title("Spectrum - Data Filtered with Filter Type " + str(j+1))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')

plt.show()
write('StarWars3_filtered.wav', Fs, f_data[6].astype(np.int16))
print("Finished!")
