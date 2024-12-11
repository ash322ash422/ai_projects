#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

vector<vector<double>> load_ratings(const string& file_path) {
    ifstream file(file_path);
    string line;
    vector<vector<double>> ratings_matrix;

    while (getline(file, line)) {
        stringstream ss(line);
        string item;
        vector<double> row;

        while (getline(ss, item, ',')) {
            row.push_back(stod(item));
        }
        ratings_matrix.push_back(row);
    }
    return ratings_matrix;
}

MatrixXd calculate_user_similarity(const MatrixXd& ratings_matrix) {
    MatrixXd ratings_matrix_nan = ratings_matrix;
    ratings_matrix_nan = ratings_matrix_nan.unaryExpr([](double val) { return val == 0 ? NAN : val; });

    MatrixXd similarity = (ratings_matrix_nan.transpose().colwise() - ratings_matrix_nan.rowwise().mean()).array().rowwise().normalized();
    return similarity.cwiseAbs();
}

MatrixXd predict_ratings(const MatrixXd& ratings_matrix, const MatrixXd& user_similarity) {
    MatrixXd predicted_ratings = ratings_matrix;
    MatrixXd user_means = ratings_matrix.rowwise().mean();

    for (int user = 0; user < ratings_matrix.rows(); ++user) {
        for (int movie = 0; movie < ratings_matrix.cols(); ++movie) {
            if (ratings_matrix(user, movie) == 0) {
                VectorXd similar_users = user_similarity.row(user);
                VectorXd ratings = ratings_matrix.col(movie);
                VectorXd ratings_centered = ratings.array() - user_means(user);

                double weighted_sum = (similar_users.array() * ratings_centered.array()).sum();
                double sum_of_weights = similar_users.array().abs().sum();

                if (sum_of_weights != 0) {
                    predicted_ratings(user, movie) = user_means(user) + (weighted_sum / sum_of_weights);
                } else {
                    predicted_ratings(user, movie) = user_means(user);
                }
            }
        }
    }
    return predicted_ratings;
}

void recommend_movies(const MatrixXd& predicted_ratings, int user_index, const vector<string>& movie_names, int n = 3) {
    VectorXd user_ratings = predicted_ratings.row(user_index);
    VectorXd original_ratings = predicted_ratings.row(user_index); // Assuming original ratings are in the same matrix for simplicity

    vector<pair<double, int>> recommendations;
    for (int i = 0; i < user_ratings.size(); ++i) {
        if (original_ratings(i) == 0) {
            recommendations.emplace_back(user_ratings(i), i);
        }
    }

    sort(recommendations.rbegin(), recommendations.rend());

    cout << "Top " << n << " recommended movies for User " << user_index << ":\n";
    for (int i = 0; i < min(n, (int)recommendations.size()); ++i) {
        cout << movie_names[recommendations[i].second] << ": " << recommendations[i].first << endl;
    }
}

double calculate_rmse(const MatrixXd& original_ratings, const MatrixXd& predicted_ratings) {
    VectorXd mask = (original_ratings.array() > 0).cast<double>();
    VectorXd mse = ((original_ratings - predicted_ratings).array().square() * mask.array()).sum();
    return sqrt(mse.sum() / mask.sum());
}

int main() {
    string ratings_file = "ratingsTest.csv";
    vector<vector<double>> ratings_data = load_ratings(ratings_file);
    MatrixXd ratings_matrix = Map<MatrixXd>(&ratings_data[0][0], ratings_data.size(), ratings_data[0].size());

    cout << "Original Ratings Matrix (first 6 entries):\n";
    for (int i = 0; i < min(6, (int)ratings_matrix.rows()); ++i) {
        cout << ratings_matrix.row(i) << endl;
    }

    MatrixXd user_similarity = calculate_user_similarity(ratings_matrix);
    cout << "\nUser Similarity Matrix:\n" << user_similarity << endl;

    MatrixXd predicted_ratings = predict_ratings(ratings_matrix, user_similarity);
    cout << "\nPredicted Ratings Matrix (first 6 entries):\n";
    for (int i = 0; i < min(6, (int)predicted_ratings.rows()); ++i) {
        cout << predicted_ratings.row(i) << endl;
    }

    double rmse = calculate_rmse(ratings_matrix, predicted_ratings);
    cout << "\nRMSE: " << rmse << endl;

    vector<string> movie_names = {"Movie1", "Movie2", "Movie3"}; // Replace with actual movie names
    recommend_movies(predicted_ratings, 0, movie_names, 3);

    return 0;
}

