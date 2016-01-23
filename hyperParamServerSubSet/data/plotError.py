import numpy as np
import pandas as pd
import pylab as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerLine2D

import os


def resultFile(fname):
    resultFile = os.path.expanduser('~/Desktop/hyper_parameter_tuning/hyperParamServerSubSet/data')
    return os.path.join(resultFile, fname)

def plotResultFile():

    index = 1

    # outputPath_part_final = resultFile("resultSheet.csv")
    # output_plot =  resultFile("result.pdf")

    outputPath_part_final = "resultSheet.csv"
    output_plot = "result.pdf"

    # df = pd.read_csv(outputPath_part_final,delimiter='\t')
    df = pd.read_csv(outputPath_part_final, error_bad_lines=False)
    # df.drop(df.columns[[250]], axis=1, inplace=True)
    df = df.ix[0:43]
    i=1
    epochnum = []
    while i<=44:
        epochnum.append(i)
        i = i+1
    i=0

    plt.ylabel('test loss')
    plt.xlabel('meta iterations')

    df_1=df[df.columns[0]]
    df_2=df[df.columns[1]]

    plt.plot(df_1, label='subsets')
    plt.plot(df_2, label='fullsets')

    plt.legend(handler_map={plt: HandlerLine2D(numpoints=4)})
    # plt.show()
    # plt.show()
    plt.savefig(output_plot)
    plt.close()




if __name__ == '__main__':
    plotResultFile()