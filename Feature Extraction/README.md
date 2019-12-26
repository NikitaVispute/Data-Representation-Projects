Feature Extraction Project

Part I

Implement several feature extraction methods that reduce the dimensionality of the data from m to 2.

Input: two files: data and labels. The data is a comma separated matrix of size n x m. Here the data
points are the rows, not the columns. The "labels" is an array of size n. The labels can only take the
values 1,2,3.

Output:
1. A comma separated file containing the n x 2 matrix of the reduced data.
2. A comma separated file containing the two vectors v1; v2 as a matrix of size 2 x m.

Programs: Read the files for the matrices XT and the labels y. Use these to calculate two vectors v1; v2,
each of size m. Compute the projections of XT on v1; v2 and use them to create the matrix D of size n x 2.

Programs that you are asked to implement:
1. pca1.py : use PCA without subtracting the mean.
2. pca2.py : use PCA with mean subtraction.
3. scatter1.py : minimize the within-class scatter.
4. scatter2.py : maximize the between-class scatter.
5. scatter3.py : maximize the ratio of between-class scatter and within-class scatter.

Try your programs on the following two datasets: irises, wine.

Part II

Feature extraction can be guided by various criteria. In this part our goal is to perform dimensionality
reduction to be followed by nearest neighbor. Specifically, training is performed with the input being the
same as in Part I. Then, given a test vector x of length m, it is reduced into two dimensions vx; vy. The
program searches for the vector in the training data that is nearest to vx; vy and returns its index.
You are asked to implement only the dimension reduction part, for two cases. In the worst case all items
in the training data are candidates for the nearest neighbor. Your goal is to try to get the data as evenly
distributed as possible. In the second case the nearest neighbor must be of a specified label.

Programs
1. reducedim1.py : reduced dimension for the worst case.
2. reducedim2.py : reduced dimension for the second case.
The arguments for reducedim1.py and reducedim2.py are the same as in Part I.

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
