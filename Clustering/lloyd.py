#Nikita Vispute
#CS 6344
#NXV170005

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 16:01:03 2019

@author: Nikita Vispute
"""


# Input: k (#clusters), r (#iterations), datafile
# k is an integer. r is an integer. Each row in datafile is a point. 
# Output: cluster number of each data points

import sys
import numpy as np
from numpy import newaxis
from numpy.random import randint
from numpy.random import seed

np.seterr(divide='ignore', invalid='ignore')

# set the random seeds to make sure your results are reproducible
#seed(1)

if len(sys.argv) != 5:
    print('usage: ', sys.argv[0], 'data_file k r output_file')
    sys.exit()

#Read inputs.
data_file = sys.argv[1]
k = int(sys.argv[2])
r = int(sys.argv[3])

# customize output file
output_file = sys.argv[4]

with open(data_file,"r") as filestream:
  points = np.loadtxt(filestream, delimiter=",")

# initial selection
def initialize_centroids(points, k):
    #returns k centroids from the initial points
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

centroids = initialize_centroids(points, k)


def closest_centroid(points, centroids):
    #returns an array containing the index to the nearest centroid for each point
    distances = np.sqrt(((points - centroids[:, newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


def move_centroids(points, closest, centroids):
    #returns the new centroids assigned from the points closest to them
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

for i in range(r):
    closest = closest_centroid(points, centroids)
    #print(closest)
    centroids = move_centroids(points, closest, centroids)
    #print(centroids)

def compute_quantization_error(points, centroids, closest):
    m = 0
    for i in range(len(points)):
        m += ((points[i] - centroids[closest[i]])**2).sum()
    print("Quantization Error is: ", round(m,4))
    return m
compute_quantization_error(points, centroids, closest)


np.savetxt(output_file, closest, delimiter=',', fmt = '%i')