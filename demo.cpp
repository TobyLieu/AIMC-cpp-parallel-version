#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mpi.h"
using namespace std;
using Eigen::MatrixXd;

int main() {
    // dataset
    vector<string> ds = {"Handwritten_fea"};

    string dsPath = "./0-dataset/";
    string resPath = "./res-lmd0/";
    vector<string> metric = {"ACC", "nmi", "Purity", "Fscore", "Precision", "Recall", "AR", "Entropy"};

    for (int dsi = 0; dsi < ds.size(); dsi++) {
        vector<int> answer = {};
        // load data & make folder
        string dataName = ds[dsi];
        cout << dataName << endl;
        string path = dsPath + dataName;
        vector<MatrixXd> X;

        // load X
        for (int t = 0; t < 6; t++) {
            ifstream in(path + "/X/X" + to_string(t) + ".txt");
            int row, col;
            in >> row >> col;
            MatrixXd Xi(row, col);
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    in >> Xi(i, j);
                }
            }
            X.push_back(Xi);
        }

        // load Y
        ifstream in(path + "/Y/Y.txt");
        int row, col;
        in >> row >> col;
        MatrixXd Y(row, col);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                in >> Y(i, j);
            }
        }

        // there is no function 'unique' in cpp, so give the k directly
        int k = 10;

        // set the output path
        // ...

        // para setting
        vector<int> d = {3 * k};  //[k 2*k 3*k]

        for (int id = 0; id < d.size(); id++) {
            // tic
            // algo_AI
            // toc
        }
    }
}