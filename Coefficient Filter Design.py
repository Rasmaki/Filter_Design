import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import convolve, freqz
import matplotlib.pyplot as plt

f, data = read("StarWars3.wav")
data_len = len(data)
Fs = f
print("Calculating...")


def plot_filter_types(filter_type, raw_data):
    filtered_data = convolve(raw_data, filter_type)
    ax.plot(abs(np.fft.fft(raw_data)))
    ax.plot(abs(np.fft.fft(filtered_data)))
    return filtered_data


filter_tp_a = np.array([1, 2, 1]) / 4
filter_tp_b = np.array([1, 3, 3, 1]) / 8
filter_tp_c = np.array([1, 4, 6, 4, 1]) / 16
filter_hp_a = np.array([1, -2, 1]) / 4
filter_hp_b = np.array([1, -3, 3, -1]) / 8
filter_hp_c = np.array([1, -4, 6, -4, 1]) / 16

filter_list = [filter_tp_a, filter_tp_b, filter_tp_c, filter_hp_a, filter_hp_b, filter_hp_c]

fig, ax = plt.subplots()
f_data = []
for i in range(len(filter_list)):
    f_data.append(plot_filter_types(filter_list[i], data))


ax.set_xscale('log')
ax.set_yscale('log')
plt.show()

write('StarWars3_filtered.wav', Fs, f_data[0].astype(np.int16))
print("Finished!")
