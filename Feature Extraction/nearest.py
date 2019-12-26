# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:10:14 2019

@author: Nikita Vispute
"""

import sys
import math
import numpy as np


def euclideanDistance(instance1, instance2, length):
    # print(instance1,instance2)
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


if __name__ == '__main__':
    arg = sys.argv
    inputfile_reduceddata = arg[1]
    inputfile_vectors = arg[2]
    inputfile_label = arg[3]
    inputfile_test_point = arg[4]
    input_testLabel = arg[5]
    outputFile_test = arg[6]

    min = sys.maxsize
    reduced_data = np.genfromtxt(inputfile_reduceddata, delimiter=',')
    vectors = np.genfromtxt(inputfile_vectors, delimiter=',')
    label = np.genfromtxt(inputfile_label, delimiter=',')
    test_data = np.genfromtxt(inputfile_test_point, delimiter=",")
    test_label=np.genfromtxt(input_testLabel,delimiter=",")
    # print(test_data)
    test_matrix = np.matrix(test_data)

    file = open(outputFile_test, "w")

    finallabel = ""
    for j in range(test_data.shape[0]):
        # print(test_matrix[j])
        # print(vectors)
        # print(vectors)
        test_dict = test_matrix[j] * vectors
        # print(test_dict)
        test = test_dict.tolist()[0]
        queriedLabel=test_label[j]
        # print("Test",test)
        for i in range(reduced_data.shape[0]):
            tempLabel = int(label[i])

            # print(reduced_data[i])
            if queriedLabel != -1:
                if tempLabel == queriedLabel:
                    temp = euclideanDistance(reduced_data[i], test, 2)
            else:
                temp = euclideanDistance(reduced_data[i], test, 2)
            if (temp < min):
                min = temp
                # finallabel = str(i)
        for k in range(reduced_data.shape[0]):
            temp=euclideanDistance(reduced_data[k],test,2)
            # print(test)
            # print(min)
            if(temp==min):
                if finallabel=="":
                    finallabel+=str(k)
                else:
                    finallabel+=","+str(k)

        file.write(str(finallabel))
        finallabel=""
        min=sys.maxsize
        file.write("\n")
    file.close()