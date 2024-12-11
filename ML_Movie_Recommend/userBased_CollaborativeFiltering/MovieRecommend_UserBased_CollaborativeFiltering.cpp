#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <map>
#include <iomanip>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Load ratings matrix from CSV file
MatrixXd load_ratings(const string& file_path, vector<string>& movie_names) {
    ifstream file(file_path);
    string line;
    vector<vector<double>> data;
    
    // Read header (movie names)
    if (getline(file, line)) {
        stringstream ss(line);
        string name;
        while (getline(ss, name, ',')) {
            movie_names.push_back(name);
        }
    }

    // Read each line of user ratings
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;
        
        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }
        data.push_back(row);
    }

    // Convert vector of vectors to Eigen::MatrixXd
    int rows = data.size();
    int cols = data[0].size();
    MatrixXd ratings(rows, cols);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            ratings(i, j) = data[i][j];

    return ratings;
}

// Calculate user similarity using Pearson correlation
MatrixXd calculate_user_similarity(const MatrixXd& ratings) {
    int users = ratings.rows();
    MatrixXd similarity = MatrixXd::Zero(users, users);

    for (int i = 0; i < users; ++i) {
        for (int j = 0; j < users; ++j) {
            if (i == j) {
                similarity(i, j) = 1;
            } else {
                VectorXd user_i = ratings.row(i);
                VectorXd user_j = ratings.row(j);

                // Mask to ignore zeros (treat them as missing values)
                Array<bool, Dynamic, 1> mask = (user_i.array() != 0) && (user_j.array() != 0);

                // If no common ratings, similarity is 0
                if (mask.count() == 0) {
                    similarity(i, j) = 0;
                    continue;
                }

                // Filter out zeros
                VectorXd valid_i(mask.count());
                VectorXd valid_j(mask.count());

                int index = 0;
                for (int k = 0; k < user_i.size(); ++k) {
                    if (mask(k)) {
                        valid_i(index) = user_i(k);
                        valid_j(index) = user_j(k);
                        ++index;
                    }
                }

                // Compute means (excluding zeros)
                double mean_i = valid_i.mean();
                double mean_j = valid_j.mean();

                // Compute centered ratings
                VectorXd centered_i = valid_i.array() - mean_i;
                VectorXd centered_j = valid_j.array() - mean_j;

                double numerator = centered_i.dot(centered_j);
                double denominator = sqrt(centered_i.squaredNorm()) * sqrt(centered_j.squaredNorm());

                if (denominator != 0) {
                    similarity(i, j) = numerator / denominator;
                } else {
                    similarity(i, j) = 0;
                }
            }
        }
    }

    return similarity;
}

MatrixXd predict_ratings(const MatrixXd& ratings_matrix, const MatrixXd& user_similarity) {
    /*
    - This function uses user-based collaborative filtering to predict ratings by
       leveraging the similarity between users.
    - Ratings are centered by subtracting the user's mean to normalize rating scales,
      ensuring fair comparisons between users.
    - Predictions are made using a weighted average of the ratings from similar users,
       where the weights are the similarity scores.
    - If no meaningful similarity exists (i.e., sum_of_weights is zero), the prediction 
      falls back to the user's mean rating.
    */
   
    // Mean rating per user (ignoring unrated movies)
    int users = ratings_matrix.rows();
    int movies = ratings_matrix.cols();
    VectorXd user_means(users);
    for (int user = 0; user < users; ++user) {
        double sum = 0;
        int count = 0;
        for (int movie = 0; movie < movies; ++movie) {
            if (ratings_matrix(user, movie) != 0) {
                sum += ratings_matrix(user, movie);
                count++;
            }
        }
        user_means(user) = (count > 0) ? sum / count : 0;
    }
    // cout << "user_means=\n" << user_means.transpose() << endl;

    // Create a matrix to store predicted ratings
    MatrixXd predicted_ratings = ratings_matrix;

    for (int user = 0; user < users; ++user) {
        for (int movie = 0; movie < movies; ++movie) {
            if (ratings_matrix(user, movie) == 0) {  // Only predict for unrated movies
                // cout << "user:" << user << ", movie:" << movie << ":" << endl;

                // Compute the weighted sum of ratings from similar users
                VectorXd similar_users = user_similarity.row(user);
                VectorXd ratings = ratings_matrix.col(movie);

                // Center the ratings by subtracting user means
                VectorXd ratings_centered = ratings - user_means;
                // cout << "..similar_users=\n" << similar_users.transpose()<< endl << ", ratings_centered.fillna(0)=\n" << ratings_centered.transpose() << endl;

                // Weighted average prediction
                double weighted_sum = (similar_users.array() * ratings_centered.array()).sum();
                double sum_of_weights = similar_users.array().abs().sum();
                // cout << "..weighted_sum=" << weighted_sum << ", sum_of_weights=" << sum_of_weights << endl;

                if (sum_of_weights != 0) {
                    double predicted_rating = user_means(user) + (weighted_sum / sum_of_weights);
                    predicted_ratings(user, movie) = predicted_rating;
                } else {
                    predicted_ratings(user, movie) = user_means(user);
                }
            }
        }
    }

    return predicted_ratings;
}

// Calculate RMSE between original and predicted ratings
double calculate_rmse(const MatrixXd& original, const MatrixXd& predicted) {
    double sum_squared_error = 0;
    int count = 0;

    for (int i = 0; i < original.rows(); ++i) {
        for (int j = 0; j < original.cols(); ++j) {
            if (original(i, j) > 0) { // Only consider rated movies
                double error = original(i, j) - predicted(i, j);
                sum_squared_error += error * error;
                count++;
            }
        }
    }

    return sqrt(sum_squared_error / count);
}

// Recommend top N movies for a user
void recommend_movies(const MatrixXd& original_ratings, const MatrixXd& predicted_ratings, int user_index, int n = 3) {
    int movies = original_ratings.cols();

    // Get the predicted ratings for the specified user
    VectorXd user_predicted_ratings = predicted_ratings.row(user_index);

    // Get the original ratings for the specified user
    VectorXd user_original_ratings = original_ratings.row(user_index);

    // Store unrated movies and their predicted ratings in a map
    map<int, double> unrated_movies;
    for (int movie = 0; movie < movies; ++movie) {
        if (user_original_ratings(movie) == 0) {
            unrated_movies[movie] = user_predicted_ratings(movie);
        }
    }

    // Sort unrated movies based on predicted ratings in descending order
    vector<pair<int, double>> sorted_recommendations(unrated_movies.begin(), unrated_movies.end());
    sort(sorted_recommendations.begin(), sorted_recommendations.end(),
         [](const pair<int, double>& a, const pair<int, double>& b) {
             return a.second > b.second;
         });

    // Output the number of recommendations
    cout << "len(recommendations): " << sorted_recommendations.size() << endl;

    // Print the top n recommended movies
    cout << "Top " << n << " recommended movies for User " << user_index + 1 << ":" << endl;
    for (int i = 0; i < min(n, static_cast<int>(sorted_recommendations.size())); ++i) {
        cout << "Movie" << sorted_recommendations[i].first + 1 << ": "
             << fixed << setprecision(2) << sorted_recommendations[i].second << endl;
    }
}

int main() {
    string file_path = "ratingsTest.csv";
    vector<string> movie_names;

    // Load ratings
    MatrixXd original_ratings = load_ratings(file_path, movie_names);

    cout << "Original Ratings Matrix (first 6 rows):\n" << original_ratings.topRows(6) << "\n";

    // Calculate user similarity
    MatrixXd user_similarity = calculate_user_similarity(original_ratings);
    cout << "\nUser Similarity Matrix:\n" << user_similarity << "\n";

    // Predict ratings
    MatrixXd predicted_ratings = predict_ratings(original_ratings, user_similarity);
    cout << "\nPredicted Ratings Matrix (first 6 rows):\n" << predicted_ratings.topRows(6) << "\n";

    // Calculate RMSE
    double rmse = calculate_rmse(original_ratings, predicted_ratings);
    cout << "\nRMSE: " << rmse << "\n";

    // Recommend top 3 movies for User 1 (index 0)
    recommend_movies(original_ratings, predicted_ratings, 0, 3);

    return 0;
}

//Following is a sample run:
/*
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

*/