#Task 1: k-NN classifier 
# - Design a k -NN classifier (k =5) for all ten genres using only the following four features; 
#       spectral rolloff mean, mfcc 1 mean, spectral centroid mean and tempo.
# - Evaluate the performance of the classification mode



from data_reader.file_to_array import data_to_array

k = 5
n_genres = 10

features = [ 
        "spectral_rolloff_mean",
        "mfcc_1_mean",
        "spectral_centroid_mean",
        "tempo"
        ]

output = data_to_array("data/GenreClassData_30s.txt", features, True, True)

print(output)



