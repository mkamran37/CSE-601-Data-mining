# CSE-601
This project deals with the implementation and execution of 5 clustering algorithms, namely:
1. K-means
2. Hierarichical Agglomerative approach
3. Density based clustering (DBSCAN)
4. Gaussian Mixture Model
5. Spectral based clustering

# The entry point for execution of the project is the file named 'main.py'
1. First install all the packages from requirements.txt
2. Run 'main.py' and follow the instructions in the terminal to execute the algorithm of choice.
    2.1 For Gaussian mixture model, to run cho.txt and iyer.txt we initialize the mean using 
        kmeans. Enter 1 when asked for "Enter 1 for kmeans initialization else Enter 0".
        To change parameters mean, covariance and pi, hardcode the data in GMM.py
3. At the input prompt, enter the filename.
    3.1 REQUIRES A MAC FOR THE FILEPATH TO BE VALID!! FOR WINDOWS, NEED TO CHANGE THE ENTIRE FILEPATH VARIABLE IN helpers.py->read_data() and many other places.
4. Make sure to keep the input file in the "Data" folder
5. For output of sample input files, check the 'Results' folder.
