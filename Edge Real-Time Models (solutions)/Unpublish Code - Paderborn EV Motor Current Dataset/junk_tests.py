import numpy as np  # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import seaborn as sns


def test_1():
    '''
    y = np.array([7, 0, 0, 0, 1, 14, 5, 8, 8, 3, 2, 0, 1, 0])
    bins = np.arange(15)
    # y, x = np.histogram(mse, bins=bins)
    # n, bins, patches = plt.hist(x=mse, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    fig, ax = plt.subplots()
    ax.plot(bins[:-1], y)
    # plt.grid(axis='y', alpha=0.75)
    plt.bar(y, performance, align='center', alpha=0.5)
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.title('Partitioned Model MSE Histogram')
    plt.show()
    '''

    mse1 = [0.38410652, 0.61260368, 0.61612243, 0.71543927, 0.58967979, 0.71200946,
            0.71571015, 8.52773226, 5.7025974, 5.47833906, 6.66589041, 6.7298585,
            8.04391648, 7.27948246, 8.31057284, 7.01957788, 5.60759952, 8.37302012,
            4.9091899, 7.16830289, 5.93145671, 12.08894204, 9.56602847, 5.92615419,
            6.02905557, 8.01557266, 9.13856086, 5.32080939, 9.32896003, 5.14651804,
            8.42349291, 7.79637997, 5.2911685, 5.51768775, 8.71849616, 10.20117499,
            6.25643807, 7.26228224, 5.48497323, 7.42068021, 5.32313329, 6.21811976,
            10.25402074, 7.19553433, 8.33505427, 7.38479836, 5.90996854, 5.24716413,
            5.4461503]
    mse0 = [585.6606]
    hist, bins, patches = plt.hist(x=[mse1, mse0], alpha=0.7, rwidth=0.85, label=['Multi-Models', 'Single-Model(EDA)'])
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.title('Models MSE')
    # plt.text(23, 45, "number of models=49")
    # maxfreq = hist.max()
    # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()
    plt.pause(1)
    print("DONE!")


def test_2(idx, mean, std):
    n_k = int(idx / 7) + 1
    n_1k = idx - 7 * (n_k - 1) + 1
    title = "(n[k]=" + str(n_k) + ", n[k-1]=" + str(n_1k) + ")\n"
    title += str(f'mean = {mean:.4f}, ') + str(f' std = {2 * std:.4f}')
    print(title)
    timestr = datetime.now().strftime('_%Y%m%d%H%M%S')
    filename = "model_" + str(idx) + timestr + ".png"
    print(filename)


test_2(17, 5.678912345, 0.0123456789)
