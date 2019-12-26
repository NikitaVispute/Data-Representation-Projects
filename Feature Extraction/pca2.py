# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 18:53:38 2019

@author: Nikita Vispute
"""
import sys
import numpy as np
from numpy import mean
from numpy import cov
from numpy.linalg import eig

#PCA With Mean Subtraction
def main(input_data, input_labels, output_vector, output_reduced_data):
    
    # read input data and labels
    
    #Xt = np.loadtxt("iris.data", delimiter =',')
    Xt = np.loadtxt(input_data, delimiter =',')
    #Xt = np.matrix(input_data)
    #print("Xt: \n", Xt) 
    
    #y = np.loadtxt("iris.labels", delimiter =',')
    y = np.loadtxt(input_labels, delimiter=',') 
    #print("y: \n", y)
    
    # calculate the mean of each column
    M = mean(Xt.T, axis=1)
    #print("Mean of each column: \n", M)
    
    # center columns by subtracting column means (centered data)
    C = Xt - M
    #print("After Mean Subtraction: \n", C)
    
    #calculate R = C.T * C (covariance matrix of centered matrix)
    #R = np.dot(C.T,C)
    #R = R/(C.shape[0]-1)
    #print("R Matrix: \n", R)
    
    # calculate covariance matrix of centered matrix
    CM = cov(C.T)
    CM = np.matrix(CM)
    #print("Covariance Matrix: \n", CM)
    
    # eigenvalue decomposition of covariance matrix
    eig_vals, eig_vecs = eig(CM)
    
    #Given no of PCA Components = 2 since Vt vector consists of 2 vectors v1, v2
    #and has dimensions 2 x m
    #also original Xt  n x m reduced to n x 2 matrix, therefore
    num_pca_components = 2 
    
    # sort eigenvalues in decreasing order
    index = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[index].real
    #print('Eigenvalues in decreasing order:\n',eig_vals)
    
    # sort eigenvectors according to same index of eigenvalues
    eig_vecs = eig_vecs[:,index].real
    #print('Eigenvectors in decreasing order of eigenvalues:\n',eig_vecs)
    
    # select the first k eigenvectors and eigenvalues 
    #(k is desired dimension of rescaled data array, k = num_of_vectors = 2 )
    E_vec = np.matrix(eig_vecs[:, :num_pca_components].real)
    E_val = np.matrix(eig_vals[ :num_pca_components].real)
    
    #Vt consists of v1, v2 vectors where each vector is a row of Vt.
    #Dimensions of Vt are 2 x m
    #print("2 Selected eigenvectors in decreasing order of eigenvalues: \n",E_vec)
    Vt = E_vec.T    
    #print("Vt: \n", Vt)
    
    # carry out the transformation on the centered data using eigenvectors
    # and return the re-scaled data and eigenvectors used
    Xt_pcafeatured = C.dot(E_vec).real
    #print("PCA feature New Data with Mean Subtraction: \n", Xt_pcafeatured)
    D = Xt_pcafeatured
        
    #output vector []
    np.savetxt(output_vector, Vt, delimiter=',',fmt='%1.9f')
    
    #output reduced data [Matrix D of n x 2 dimensions]
    np.savetxt(output_reduced_data, D, delimiter=',',fmt='%1.9f')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])