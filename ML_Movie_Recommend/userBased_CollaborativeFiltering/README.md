# Vyorius Test: Movie Recommendation System using User-Based Collaborative Filtering
This project implements a user-based collaborative filtering recommendation system in both Python and C++. The system suggests movies to users based on how others with similar interests have rated those movies.

## step 1:

The initial design and testing were done using Python 3.11 due to its rapid development capabilities. Following libraries were used: jupyter==1.1.1, scikit-learn==1.6.0, pandas==2.2.3. As per the task requirements, I used "user-based collaborative-filtering". I tested on file ratingsTest.csv that contains 9 users and 7 movies. I was satisfied with the results. A sample run is appended at the end of the file MovieRecommend_UserBased_CollaborativeFiltering.cpp

NOTE: I did not use any Machine Learning or AI tools, although it can be used. Specifically, K-Nearest Neighbors (KNN) would be better option. I kept it simple to show proof of concept.

## step 2:

I converted the above python code into C++ code. I debugged and checked the output to make sure it matches with the python code.

# Setup Instructions

* I used the GNU g++ 13.2.0 compiler as distributed by the MSYS2 project on a Windows OS.
* I downloaded eigen-3.4.0.zip file from https://eigen.tuxfamily.org/index.php?title=Main_Page . This was needed to compile the code that used "#include <Eigen/Dense>" directive. I extracted the zip file and put the directory eigen-3.4.0 inside the projects directory.
* To compile: g++ -I path\to\directory\eigen-3.4.0 MovieRecommend_UserBased_CollaborativeFiltering.cpp -o MovieRecommend
* To execute: ./MovieRecommend.exe

* A sample run:
```
C:\ai_projects\ML_Movie_Recommend\userBased_CollaborativeFiltering>g++ -I temp\eigen-3.4.0 MovieRecommend_UserBased_CollaborativeFiltering.cpp -o MovieRecommend

C:\ai_projects\ML_Movie_Recommend\userBased_CollaborativeFiltering>MovieRecommend.exe
Original Ratings Matrix (first 6 rows):
0 5 3 0 0 2 0
0 0 0 5 3 4 1
3 4 5 4 3 3 4
3 1 5 4 4 0 5
2 4 3 1 3 0 4
4 0 4 3 0 0 1

User Similarity Matrix:
         1          0   0.327327         -1          1          0   0.944911         -1          1
         0          1  -0.169031  -0.866025  -0.981981          1         -1   0.478091          1
  0.327327  -0.169031          1   0.294118   0.189389          0          0 -0.0714286  -0.311805
        -1  -0.866025   0.294118          1  -0.151511  -0.492366  -0.866025  -0.301511  -0.868599
         1  -0.981981   0.189389  -0.151511          1  -0.547723   0.866025  -0.622543  -0.356348
         0          1          0  -0.492366  -0.547723          1         -1  -0.327327   0.755929
  0.944911         -1          0  -0.866025   0.866025         -1          1        0.5   0.654654
        -1   0.478091 -0.0714286  -0.301511  -0.622543  -0.327327        0.5          1         -1
         1          1  -0.311805  -0.868599  -0.356348   0.755929   0.654654         -1          1

Predicted Ratings Matrix (first 6 rows):
2.99337       5       3 1.45123 2.35564       2 2.54091
3.56989 2.27647 2.63136       5       3       4       1
      3       4       5       4       3       3       4
      3       1       5       4       4 3.93396       5
      2       4       3       1       3 2.36317       4
      4 2.05311       4       3 2.81514  2.9661       1

RMSE: 0
len(recommendations): 4
Top 3 recommended movies for User 1:
Movie1: 2.99
Movie7: 2.54
Movie5: 2.36

```

