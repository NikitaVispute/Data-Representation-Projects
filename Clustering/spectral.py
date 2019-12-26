#Nikita Vispute
#CS 6344
#NXV170005

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 19:03:07 2019

@author: Nikita Vispute
"""

import sys
import numpy as np
import math
from numpy.random import seed

# set the random seeds to make sure your results are reproducible
#seed(1)

def find_laplacian(X,rows,sigma):
	data_size = len(rows)
	distance=np.zeros((data_size,data_size))
	for i in range(data_size):
		for j in range(data_size):
			distance[i,j]=np.linalg.norm(X[i]- X[j])
	
	weight = np.zeros((data_size,data_size))
	degree = np.zeros(data_size)
	for i in range(data_size):
		for j in range(data_size):
			gamma = 0.5*math.pow((distance[i,j]/sigma),2)
			weight[i,j] = math.exp(-gamma);
			degree[i] = degree[i] + weight[i,j]
	
	norm_laplacian = np.zeros((data_size,data_size))
	for i in range(data_size):
		for j in range(data_size):
			if i==j and degree[i]!=0:
				norm_laplacian[i,j]=1
			elif i!=j and degree[i]!=0 and degree[j]!=0:  
				norm_laplacian[i,j]=(-weight[i,j])/math.sqrt(degree[i]*degree[j])
			else:
				norm_laplacian[i,j]=0
	return weight,degree,norm_laplacian

def spectral_clustering(rows,weight,degree,norm_laplacian):
	data_size = len(rows)
	eigen_vals, eigen_vecs = np.linalg.eig(norm_laplacian)
	#need to select top 2 eigen values and eigen vectors
	idx = np.argsort(eigen_vals)[::-1] # sort in reverse order
	eigen_vals = eigen_vals[idx]
	eigen_vecs = eigen_vecs[:,idx]
	eigen_val_second_small = eigen_vals[-2]
	eigen_vec_second_small = eigen_vecs[:,-2]
	idx = np.argsort(eigen_vec_second_small)[::]
	h=float('inf')
	part1 = None
	part2 = None
	for k in range(len(idx)-1):
		A = set(idx[:k+1])
		B = set(idx[k+1:])
		Va = 0
		Vb = 0
		Cab = 0
		for i in range(data_size):
			if i in A:
				Va=Va+degree[i]
			elif i in B:
				Vb=Vb+degree[i]
			for j in range(data_size):
				if i in A and j in B:
					Cab=Cab+weight[i,j]
		temp = Cab/(float(min(Va,Vb)))
		if temp<h:
			part1 = list(A)
			part2 = list(B)
			h=temp
	
	part1 = np.asarray(rows)[part1].tolist()
	part2 = np.asarray(rows)[part2].tolist()
	return part1,part2
	
def quantization_error(X,part_list):
	error_value=0.0
	for part in part_list:
		data = X[part]
		centroid=data.mean(axis=0)
		for row in data:
			error_value=error_value+math.pow(np.linalg.norm(row- centroid),2)
		
	return (round(error_value,4))

if len(sys.argv) != 5:
    print('usage: ', sys.argv[0], 'data_file k sigma output_file')
    sys.exit()

#Read inputs.
data_file = sys.argv[1]
k = int(sys.argv[2])
sigma = float(sys.argv[3])
 
# customize output file
output_file = sys.argv[4]
  
X = np.genfromtxt(data_file, delimiter=',', autostrip=True)
part_list = [list(i for i in range(X.shape[0]))]
part = part_list.pop(0)
weight,degree,norm_laplacian = find_laplacian(X,part,sigma)
new_partitons = spectral_clustering(part,weight,degree,norm_laplacian)
part_list.extend(new_partitons)

while len(part_list)<k:
	min_lambda=float('inf')
	min_partition=None
	min_weight=None
	min_degree=None
	min_norm_laplacian=None
	for K in range(len(part_list)):
		weight,degree,norm_laplacian = find_laplacian(X,part_list[K],sigma)
		eigen_vals, eigen_vecs = np.linalg.eig(norm_laplacian)
		idx = np.argsort(eigen_vals)[::-1] # sort in reverse order
		eigen_vals = eigen_vals[idx]
		eigen_val_second_small = eigen_vals[-2]
		if eigen_val_second_small<min_lambda:
			min_degree=degree
			min_norm_laplacian = norm_laplacian
			min_partition = K
			min_weight = weight
	partition = part_list.pop(min_partition)
	new_part = spectral_clustering(partition, min_weight, min_degree, min_norm_laplacian)
	part_list.extend(new_part)

cluster_mapping=np.zeros((X.shape)[0])
for i in range(len(part_list)):
	for j in part_list[i]:
		cluster_mapping[j]=i
		

np.savetxt(output_file, np.array(cluster_mapping), delimiter=',',fmt='%d')
	
print("Quantization Error is : ",quantization_error(X,part_list))