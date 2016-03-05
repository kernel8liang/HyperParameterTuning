__author__ = 'yumeng'

import os
import time
import logging
import re

from os import listdir
from os.path import isfile, join


def main():
    inputDir = os.path.realpath("/home/yumeng/Desktop/random/hyperopt_experiment_new_1/output")
    outputPath = os.path.realpath("/home/yumeng/Documents/output.txt")

    with open(outputPath, 'w') as fout:
        for f in listdir(inputDir):
            if isfile(join(inputDir, f)):
                filePath = join(inputDir, f)
                ouputLine = ""
                result = ""
                with open(filePath, 'r') as fin:
                    for line in fin:
                        if 'INFO:neon.models.mlp:epoch: ' in line:
                            ouputLine = ouputLine + line.split()[4] + "\t"
                        elif 'Got result ' in line:
                            result = line.split()[2] + "\t"
                    print(f.split(".out")[0] + "\t" + result + ouputLine + "\n")
                    fout.write(f.split(".out")[0] + "\t" + result + ouputLine + "\n")


def readErrorNeon():
    experiment_number = "120"
    inputDir_full = os.path.realpath(
        "/home/yumeng/Desktop/random/hyperopt_experiment_new_" + experiment_number + "/output")
    outputPath_full = os.path.realpath("/home/yumeng/Desktop/newLearningError/output_new_" + experiment_number + ".txt")

    with open(outputPath_full, 'w') as fout:
        for f in listdir(inputDir_full):
            if isfile(join(inputDir_full, f)):
                filePath = join(inputDir_full, f)
                ouputLine = ""
                result = ""
                with open(filePath, 'r') as fin:
                    for line in fin:
                        if 'INFO:neon.models.mlp:epoch: ' in line:
                            ouputLine = ouputLine + line.split()[4] + "\t"
                        elif 'spear_wrapper params are:{' in line:
                            hyperparameter = line.replace("\t", " ").replace(",", " ").strip()
                    # print(hyperparameter + "\t" + ouputLine + "\n")
                    fout.write(hyperparameter + "\t" + ouputLine + "\n")

    inputDir_part = os.path.realpath(
        "/home/yumeng/Desktop/random_sub/hyperopt_experiment_" + experiment_number + "/output")
    outputPath_part = os.path.realpath("/home/yumeng/Desktop/newLearningError/output_old_" + experiment_number + ".txt")

    with open(outputPath_part, 'w') as fout:
        for f in listdir(inputDir_part):
            if isfile(join(inputDir_part, f)):
                filePath = join(inputDir_part, f)
                ouputLine = ""
                result = ""
                with open(filePath, 'r') as fin:
                    for line in fin:
                        if 'INFO:neon.models.mlp:epoch: ' in line:
                            ouputLine = ouputLine + line.split()[4] + "\t"
                        elif 'spear_wrapper params are:{' in line:
                            hyperparameter = line.replace("\t", " ").replace(",", " ").strip()
                    # print(hyperparameter + "\t" + ouputLine + "\n")
                    fout.write(hyperparameter + "\t" + ouputLine + "\n")

    outputPath_part_final = os.path.realpath(
        "/home/yumeng/Desktop/newLearningError/output_final_" + experiment_number + ".txt")

    count = 0
    i = 0
    j = 0
    listfull = ""
    listpart = ""
    while i<100:
        if i<60:

            listpart=listpart+"\t"+"partdata_"+str(i)
        listfull= listfull+"\t"+"fulldata_"+str(i)
        i = i+1
    with open(outputPath_part_final, 'w') as fout:

        # fout.write("alpha2"+"\t"+"alpha1"+"\t"+"ksize2"+"\t"+"ksize1"+"\t"+"coef"+"\t"+"lr"+"\t"+"beta2"+"\t"+"beta1"+listfull+listpart+"\n")
        with open(outputPath_full, 'r') as fin_full:
            for line_full in fin_full:
                if count<400:
                    flag = False
                    print "start job number " + str(count)
                    with open(outputPath_part, 'r') as fin_part:
                        for line_part in fin_part:
                            if flag ==False:
                                split_full = line_full.split("\t")
                                split_part = line_part.split("\t")
                                hyper_full = split_full[0]
                                hyper_part = split_part[0]
                                if hyper_full == hyper_part:
                                    print"find one of the hyperparameter"
                                    flag = True
                                    hyperlist = hyper_full.split("([")
                                    alpha2 = hyperlist[1].split("])")[0]
                                    alpha1 = hyperlist[2].split("])")[0]
                                    ksize2 = hyperlist[3].split("])")[0]
                                    ksize1 = hyperlist[4].split("])")[0]
                                    coef = hyperlist[5].split("])")[0]
                                    lr = hyperlist[6].split("])")[0]
                                    beta2 = hyperlist[7].split("])")[0]
                                    beta1 = hyperlist[8].split("])")[0]

                                    removefirst_full = "\t".join(split_full[1:])
                                    removefirst_part = "\t".join(split_part[1:])

                                    # outputline = str(count)+"\t"+alpha2 + "\t" + alpha1 + "\t" + ksize2 + "\t" + ksize1 + "\t" + coef + "\t" + lr + "\t" + beta2 + "\t" + beta1 + "\t" + removefirst_full.strip() + "\t" + removefirst_part
                                    outputline = alpha2 + "\t" + alpha1 + "\t" + ksize2 + "\t" + ksize1 + "\t" + coef + "\t" + lr + "\t" + beta2 + "\t" + beta1 + "\t" + removefirst_full.strip() + "\t" + removefirst_part
                                    fout.write(outputline)
                                    count = count + 1
                            else:
                                break
                else:
                    return

def readErrorKeras():
    experiment_number = 5
    for index in range(1,experiment_number+1):
        inputDir_full = os.path.realpath(
            "/home/jie/docker_folder/random_keras/output_cifar10_mlp/hyperopt_experiment_output" + str(index) + "/output")
        outputPath_full = os.path.realpath("/home/jie/docker_folder/random_keras/output_cifar10_mlp/hyperopt_experiment_withoutparam_accuracy" + str(index) + ".txt")

        with open(outputPath_full, 'w') as fout:
            for f in listdir(inputDir_full):
                if isfile(join(inputDir_full, f)):
                    filePath = join(inputDir_full, f)
                    ouputLine = ""
                    result = ""
                    count = 0
                    with open(filePath, 'r') as fin:
                        for line in fin:
                            if 'val_loss: ' in line:
                                splict = line.replace("val_loss: ","\t").replace("- val_acc: ","\t").split("\t")
                                ouputLine = ouputLine + splict[2].replace("\n","") + "\t"
                                count = count +1
                            elif 'spear_wrapper params are:{' in line:
                                hyperparameter = line.replace("\t", " ").replace(",", " ").strip()
                        # print(hyperparameter + "\t" + ouputLine + "\n")
                        # fout.write(hyperparameter + "\t" + ouputLine + "\n")
                        if count == 600:
                            fout.write( ouputLine + "\n")

def readErrorKerasCifar100():
    experiment_number = 1
    for index in range(1,experiment_number+1):
        inputDir_full = os.path.realpath(
            "/Users/yumengyin/Documents/FYP/sem2/experiment/experiment_CIFAR10_BNN/train/")
        outputPath_full = os.path.realpath("/Users/yumengyin/Documents/FYP/sem2/experiment/error_bnn.txt")
        outputPath_full1 = os.path.realpath("/Users/yumengyin/Documents/FYP/sem2/experiment/accu_bnn.txt")

        with open(outputPath_full, 'w') as fout:
            with open(outputPath_full1, 'w') as fout1:
                for f in listdir(inputDir_full):
                    if isfile(join(inputDir_full, f)):
                        filePath = join(inputDir_full, f)
                        ouputLine = ""
                        ouputLine1= ""
                        result = ""
                        count = 0
                        first= True
                        with open(filePath, 'r') as fin:
                            for line in fin:
                                if first==True:
                                    hyperparameter = line.replace("\n","")
                                    first=False
                                if 'val_loss: ' in line:
                                    splict = line.replace("val_loss: ","\t").replace("- val_acc: ","\t").split("\t")

                                    ouputLine = ouputLine + splict[1].replace("\n","") + "\t"
                                    ouputLine1 = ouputLine1 + splict[2].replace("\n","") + "\t"
                                    count = count +1
                                # elif 'spear_wrapper params are:{' in line:
                                #     hyperparameter = line.replace("\t", " ").replace(",", " ").strip()
                            
                            # fout.write(hyperparameter + "\t" + ouputLine + "\n")
                            # if count == 99:
                        fout.write(hyperparameter+"\t"+ ouputLine + "\n")
                        fout1.write(hyperparameter+"\t"+ ouputLine1 + "\n")

                        # print(hyperparameter + "\t" + ouputLine + "\n")
        inputDir_full = os.path.realpath(
            "/Users/yumengyin/Documents/FYP/sem2/experiment/experiment_CIFAR10_GP/train/")
        outputPath_full = os.path.realpath("/Users/yumengyin/Documents/FYP/sem2/experiment/error_gp.txt")
        outputPath_full1 = os.path.realpath("/Users/yumengyin/Documents/FYP/sem2/experiment/accu_gp.txt")

        with open(outputPath_full, 'w') as fout:
            with open(outputPath_full1, 'w') as fout1:
                for f in listdir(inputDir_full):
                    if isfile(join(inputDir_full, f)):
                        filePath = join(inputDir_full, f)
                        ouputLine = ""
                        ouputLine1= ""
                        result = ""
                        count = 0
                        first= True
                        with open(filePath, 'r') as fin:
                            for line in fin:
                                if first==True:
                                    hyperparameter = line.replace("\n","")
                                    first=False
                                if 'val_loss: ' in line:
                                    splict = line.replace("val_loss: ","\t").replace("- val_acc: ","\t").split("\t")

                                    ouputLine = ouputLine + splict[1].replace("\n","") + "\t"
                                    ouputLine1 = ouputLine1 + splict[2].replace("\n","") + "\t"
                                    count = count +1
                                # elif 'spear_wrapper params are:{' in line:
                                #     hyperparameter = line.replace("\t", " ").replace(",", " ").strip()
                            
                            # fout.write(hyperparameter + "\t" + ouputLine + "\n")
                            # if count == 99:
                        fout.write(hyperparameter+"\t"+ ouputLine + "\n")
                        fout1.write(hyperparameter+"\t"+ ouputLine1 + "\n")
def readErrorKerasCifar100_smallest():
    experiment_number = 1
    for index in range(1,experiment_number+1):
        inputDir_full = "BNN/train"
        outputPath_full = "error_bnn.txt"
        outputPath_full1 = "accu_bnn.txt"

        with open(outputPath_full, 'w') as fout:
            with open(outputPath_full1, 'w') as fout1:
                for f in listdir(inputDir_full):
                    if isfile(join(inputDir_full, f)):
                        filePath = join(inputDir_full, f)
                        ouputLine = ""
                        ouputLine1= ""
                        result = ""
                        count = 0
                        first= True
                        with open(filePath, 'r') as fin:
                            for line in fin:
                                if first==True:
                                    hyperparameter = line.replace("\n","")
                                    first=False
                                if 'val_loss: ' in line:
                                    splict = line.replace("val_loss: ","\t").replace("- val_acc: ","\t").split("\t")

                                    ouputLine = ouputLine + splict[1].replace("\n","") + "\t"
                                    ouputLine1 = ouputLine1 + splict[2].replace("\n","") + "\t"
                                    count = count +1
                                # elif 'spear_wrapper params are:{' in line:
                                #     hyperparameter = line.replace("\t", " ").replace(",", " ").strip()
                            
                            # fout.write(hyperparameter + "\t" + ouputLine + "\n")
                            # if count == 99:
                        fout.write(hyperparameter+"\t"+ ouputLine + "\n")
                        fout1.write(hyperparameter+"\t"+ ouputLine1 + "\n")

                        # print(hyperparameter + "\t" + ouputLine + "\n")
        inputDir_full = "GP/train"
        outputPath_full = "error_gp.txt"
        outputPath_full1 = "accu_gp.txt"

        with open(outputPath_full, 'w') as fout:
            with open(outputPath_full1, 'w') as fout1:
                for f in listdir(inputDir_full):
                    if isfile(join(inputDir_full, f)):
                        filePath = join(inputDir_full, f)
                        ouputLine = ""
                        ouputLine1= ""
                        result = ""
                        count = 0
                        first= True
                        with open(filePath, 'r') as fin:
                            for line in fin:
                                if first==True:
                                    hyperparameter = line.replace("\n","")
                                    first=False
                                if 'val_loss: ' in line:
                                    splict = line.replace("val_loss: ","\t").replace("- val_acc: ","\t").split("\t")

                                    ouputLine = ouputLine + splict[1].replace("\n","") + "\t"
                                    ouputLine1 = ouputLine1 + splict[2].replace("\n","") + "\t"
                                    count = count +1
                                # elif 'spear_wrapper params are:{' in line:
                                #     hyperparameter = line.replace("\t", " ").replace(",", " ").strip()
                            
                            # fout.write(hyperparameter + "\t" + ouputLine + "\n")
                            # if count == 99:
                        fout.write(hyperparameter+"\t"+ ouputLine + "\n")
                        fout1.write(hyperparameter+"\t"+ ouputLine1 + "\n")

if __name__ == '__main__':
    # main()
    readErrorKerasCifar100_smallest()