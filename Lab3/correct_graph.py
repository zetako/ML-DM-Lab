#!/bin/python
import numpy as np
import matplotlib.pyplot as plt

def draw_plot (title, *args):
    y_arr = []
    for npy in args:
        y_arr.append(np.load(npy))
    x = np.arange(1, len(y_arr[0])+1)
    color_arr = ["red", "green", "blue", "skyblue", "purple"]
    for index in range(len(y_arr)):
        plt.plot(x, y_arr[index], color = color_arr[index], label = args[index])
    plt.title(title)
    plt.legend()
    plt.show()

draw_plot('3 Models', 'history/SGD.npy', 'history/normal.npy', 'history/LeNet.npy')
draw_plot('LC training', 'history/SGD.npy', 'history/SGDM.npy', 'history/Adam.npy')
draw_plot('MLP training', 'history/normal.npy', 'history/plus.npy')
draw_plot('CNN training', 'history/LeNet.npy', 'history/LeNet_Add_Conv.npy', 'history/LeNet_Avg_Pool.npy', 'history/LeNet_Large_Filter.npy')
