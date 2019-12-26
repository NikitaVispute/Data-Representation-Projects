The folder consists of the following python codes:

1. pca1.py
2. pca2.py
3. scatter1.py
4. scatter2.py
5. scatter3.py
6. reducedim1.py
7. reducedim2.py


All the python codes are to be executed from the command line with 4 arguments
eg. python pca1.py [iris.data] [iris.labels] [output_vector] [output_reduced_data]

The last program: reducedim2.py takes 1 more argument - queried label
eg. python reducedim2.py [iris.data] [iris.labels] [output_vector] [output_reduced_data] [queried_label]

queried label can be 1,2,3 so that the reduced data will show only those candidates for nearest neighbor having the same class as the queried class.
