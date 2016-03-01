import numpy as np
import pandas as pd
import pylab as plt

import os

def plotNeonExperiment():
    experiment_number = "120"
    outputPath_part_final = os.path.realpath(
            "/home/yumeng/Desktop/newLearningError/output_final_" + experiment_number + ".txt")
    output_plot_original = os.path.realpath(
            "/home/yumeng/Desktop/newLearningError/output_plot_original_" + experiment_number + ".png")
    output_plot_full = os.path.realpath(
            "/home/yumeng/Desktop/newLearningError/output_plot_full_" + experiment_number + ".png")
    df = pd.read_csv(outputPath_part_final,delimiter='\t',header=None)
    df.drop(df.columns[[168]], axis=1, inplace=True)

    epochfull = []
    epochpart = []
    i=1
    while i<=100:
            if i<=60:

                epochfull.append(i)
            epochpart.append(i)
            i = i+1

    print epochpart
    # i=0
    # while i<400:
    #     df_1=df[df.columns[68:168]].ix[i]
    #     np.reshape(df_1, (1,100))
    #     plt.plot(epochpart,df_1)
    #
    #     i = i+1
    # plt.savefig(output_plot_original)
    # plt.close()
    # # # plt.show()
    #
    # i=0
    # while i<=399:
    #     df_2=df[df.columns[8:68]].ix[i]
    #     np.reshape(df_2, (1,60))
    #     plt.plot(epochfull,df_2)
    #
    #     i = i+1
    # plt.savefig(output_plot_full)
    # # plt.show()


    i=0
    while i<50:
        df_1=df[df.columns[68:168]].ix[i]
        np.reshape(df_1, (1,100))
        plt.plot(epochpart,df_1)

        i = i+1
    plt.savefig(output_plot_original)
    plt.close()
    # # plt.show()

    i=0
    while i<=50:
        df_2=df[df.columns[8:68]].ix[i]
        np.reshape(df_2, (1,60))
        plt.plot(epochfull,df_2)

        i = i+1
    plt.savefig(output_plot_full)
    # plt.show()

def plotKerasExperimentcifar10():

    index = 5
    for experiment_number in range(1,index+1):
        outputPath_part_final = os.path.realpath( "/home/jie/docker_folder/random_keras/output_cifar10_mlp/errorFile/hyperopt_experiment_withoutparam_accuracy" + str(experiment_number) + ".txt")
        output_plot = os.path.realpath(
                "/home/jie/docker_folder/random_keras/output_cifar10_mlp/errorFile/plotErrorCurve" + str(experiment_number) + ".pdf")

        df = pd.read_csv(outputPath_part_final,delimiter='\t',header=None)
        df.drop(df.columns[[600]], axis=1, inplace=True)

        i=1
        epochnum = []
        while i<=250:
            epochnum.append(i)
            i = i+1
        i=0
        while i<10:
            df_1=df[df.columns[0:250]].ix[i]
            np.reshape(df_1, (1,250))
            plt.plot(epochnum,df_1)

            i = i+1
        # plt.show()
        # plt.show()
        plt.savefig(output_plot)
        plt.close()

    #
    # i=0
    # while i<=399:
    #     df_2=df[df.columns[8:68]].ix[i]
    #     np.reshape(df_2, (1,60))
    #     plt.plot(epochfull,df_2)
    #
    #     i = i+1
    # plt.savefig(output_plot_full)
    # # plt.show()

def plotKerasExperimentcifar100():

    index = 8
    dir="/Users/yumengyin/Desktop/accuracy/"
    for experiment_number in range(1,index+1):
        outputPath_part_final = os.path.realpath( dir+"accu_full_sub_"+str(experiment_number)+".txt")


#first 40 row full, next 60 row sub
        full_num=40
        sub_num=60
        total =full_num+sub_num
        df = pd.read_csv(outputPath_part_final,delimiter='\t',header=None)
        df.drop(df.columns[[full_num+sub_num]], axis=1, inplace=True)
        
        

        i=1
        epochnum = []
        while i<=total:
            epochnum.append(i)
            i = i+1
        i=0
#        while i<50:
#            i = i+1
        for num in range(1,10):
            output_plot = os.path.realpath(dir+"plotAccuCurve_"+str(experiment_number)+"_"+str(num)+".pdf")
            while i<num*10:
                df_1=df[df.columns[0:total]].ix[i]
                np.reshape(df_1, (1,total))
                plt.plot(epochnum,df_1)

                i = i+1

            # plt.show()
    #        plt.show()
            plt.savefig(output_plot)
            plt.close()

def plotKerasExperimentcifar10():

    index = 8
    dir="/Users/yumengyin/Documents/FYP/sem2/experiment/error.txt"
    for experiment_number in range(1,index+1):
        # outputPath_part_final = os.path.realpath( dir+"accu_full_sub_"+str(experiment_number)+".txt")
        outputPath_part_final = "/Users/yumengyin/Documents/FYP/sem2/experiment/error.txt"
        output_plot = os.path.realpath(
                "/Users/yumengyin/Documents/FYP/sem2/experiment/error.pdf")

        df = pd.read_csv(outputPath_part_final,delimiter='\t')
        # df.drop(df.columns[[200]], axis=1, inplace=True)

        i=1
        epochnum = []
        while i<=197:
            epochnum.append(i)
            i = i+1
        # i=0
        # while i<2:
        df_1= df[df.columns[0:1]]
        print(df_1.shape)
            # np.reshape(df_1, (1,197))
        plt.plot(epochnum,df_1)
            # df_1=df[df.columns[0:250]].ix[i]
            # np.reshape(df_1, (1,250))
            # plt.plot(epochnum,df_1)
        df_1= df[df.columns[0:2]]
        print(df_1.shape)
            # np.reshape(df_1, (1,197))
        plt.plot(epochnum,df_1)
            # i = i+1
        # plt.show()
        # plt.show()
        plt.savefig(output_plot)
        plt.close()
if __name__ == '__main__':
    # main()
    plotKerasExperimentcifar10()