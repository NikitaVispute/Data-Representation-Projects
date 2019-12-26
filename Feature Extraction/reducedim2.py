# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:53:41 2019

@author: Nikita Vispute
"""

import numpy as np
import sys
from numpy import cov
from numpy import mean
from numpy.linalg import eig

def main(input_data, input_labels, output_vector, output_reduced_data, queried_label):
    
    # read input data and labels

    #input_data = np.loadtxt("iris.data", delimiter =',')
    Xt = np.loadtxt(input_data, delimiter =',')
    #Xt = np.matrix(input_data)
    #print("Xt: \n", Xt) 
    
    #y = np.loadtxt("iris.labels", delimiter =',')
    y = np.loadtxt(input_labels,  delimiter=',') 
    #print("y: \n", y)
    
    # calculate the mean of each column
    M = mean(Xt.T, axis=1)
    #print("Mean of each column: \n", M)
    
    # center columns by subtracting column means (centered data)
    C = Xt - M
    #print("After Mean Subtraction: \n", C)
    
    # calculate covariance matrix of centered matrix
    cov_matrix = cov(C.T)
    cov_matrix = np.matrix(cov_matrix)
    #print("Covariance Matrix: \n", CM)
    
    #eigenvalues and vectors of covariance matrix
    eig_vals, eig_vecs = eig(cov_matrix)
    
    #Given no of PCA Components = 2 since Vt vector consists of 2 vectors v1, v2
    #and has dimensions 2 x m
    #also original Xt  n x m reduced to n x 2 matrix, therefore
    num_vectors = 2 

    # sort eigenvalues in decreasing order
    index = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[index].real
    #print('Eigenvalues in decreasing order:\n',eig_vals)
    
    # sort eigenvectors according to same index of eigenvalues
    eig_vecs = eig_vecs[:,index].real
    #print('Eigenvectors in decreasing order of eigenvalues:\n',eig_vecs)
    
    # select the first k eigenvectors and eigenvalues 
    #(k is desired dimension of rescaled data array, k = num_of_vectors = 2 )
    E_vec = np.matrix(eig_vecs[:, :num_vectors].real)
    E_val = np.matrix(eig_vals[ :num_vectors].real)
    
    #Vt consists of v1, v2 vectors where each vector is a row of Vt.
    #Dimensions of Vt are 2 x m
    #print("2 Selected eigenvectors in decreasing order of eigenvalues: \n",E_vec)
    Vt = E_vec.T    
    #print("Vt: \n", Vt)
    
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data and eigenvectors used
    Xt_reduced = C.dot(E_vec).real
    #print("New Reduced Data: \n", Xt_reduced)
    D = Xt_reduced
    
    final_D = []
    for i in range(D.shape[0]):
        # print("label= ",int(label[i])," queried label= ",queriedLabel)
        if int(y[i]) == int(queried_label):
            final_D.append(D[i].tolist()[0])

    #output vector []
    np.savetxt(output_vector, Vt, delimiter=',',fmt='%1.9f')
        
    #output reduced data [Matrix D of n x 2 dimensions]
    np.savetxt(output_reduced_data, final_D, delimiter=',',fmt='%1.9f')
    

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])