# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:29:32 2019

@author: Nikita Vispute
"""

import sys
import numpy as np
from numpy import mean
from numpy.linalg import eig

def main(input_data, input_labels, output_vector, output_reduced_data):
    
    # read input data and labels
    
    #input_data = np.loadtxt("iris.data", delimiter =',')
    input_data = np.loadtxt(input_data, delimiter =',')
    Xt = np.matrix(input_data)
    #print("Xt: \n", Xt) 
    
    #y = np.loadtxt("iris.labels", delimiter =',')
    y = np.loadtxt(input_labels,  delimiter=',') 
    #print("y: \n", y)
    
    # y (labels) has 3 classes in each dataset - 1,2,3
    
    #calculating mean of corresponding data points in Xt for each class in y
    mean_vectors = []
    for class_val in range(1,4):
        mean_vectors.append(mean(Xt[y==class_val], axis=0))
    #    print('Mean Vector class %s: %s\n' %(class_val, mean_vectors[class_val-1]))
        
    #Computing Within-class Scatter Matrix
    W_SM = np.zeros((Xt.shape[1],Xt.shape[1]))
    for class_val,mv in zip(range(1,4), mean_vectors):
        class_sc_mat = np.zeros((Xt.shape[1],Xt.shape[1]))   # scatter matrix for every class
        for row in Xt[y == class_val]:
            row, mv = row.reshape(Xt.shape[1],1), mv.reshape(Xt.shape[1],1) # make column vectors
            class_sc_mat += (row-mv).dot((row-mv).T)
        W_SM += class_sc_mat                             # sum class scatter matrices
    #print('Within-Class Scatter Matrix:\n', W_SM)
    
    # eigenvalue decomposition of Within-class scatter matrix
    eig_vals, eig_vecs = eig(W_SM)
    
    #Given Vt vector consists of 2 vectors v1, v2
    #and has dimensions 2 x m
    #also original Xt  n x m reduced to n x 2 matrix, therefore
    num_of_vectors = 2 
    
    # sort eigenvalues in increasing order
    # to minimize Within-Class Scatter
    index = np.argsort(eig_vals)[::]
    eig_vals = eig_vals[index].real
    #print('Eigenvalues in increasing order:\n',eig_vals)
    
    # sort eigenvectors according to same index of eigenvalues
    eig_vecs = eig_vecs[:,index].real
    #print('Eigenvectors in increasing order of eigenvalues:\n',eig_vecs)
    
    # select the first k eigenvectors and eigenvalues 
    #(k is desired dimension of rescaled data array, k = num_of_vectors = 2 )
    E_vec = np.matrix(eig_vecs[:, :num_of_vectors].real)
    E_val = np.matrix(eig_vals[ :num_of_vectors].real)
    
    #Vt consists of v1, v2 vectors where each vector is a row of Vt.
    #Dimensions of Vt are 2 x m
    #print("2 Selected eigenvectors in increasing order of eigenvalues: \n",E_vec)
    Vt = E_vec.T    
    #print("Vt: \n", Vt)
    
    
    #projections of Xt on v1,v2 (eigenvectors corresponding to 2 smallest eigenvalues in incr. order)
    #this minimizes the within-class scatter
    Xt_projected = Xt.dot(E_vec).real
    #print("Projected Data: \n", Xt_projected)
    D = Xt_projected
    
    #output vector []
    np.savetxt(output_vector, Vt, delimiter=',',fmt='%1.9f')
        
    #output reduced data [Matrix D of n x 2 dimensions]
    np.savetxt(output_reduced_data, D, delimiter=',',fmt='%1.9f')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])