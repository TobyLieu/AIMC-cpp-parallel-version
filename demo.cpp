#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mpi.h"
using namespace std;

int main() {
    // dataset
    vector<string> ds = {"Handwritten_fea"};

    string dsPath = "./0-dataset/";
    string resPath = "./res-lmd0/";
    vector<string> metric = {"ACC", "nmi", "Purity", "Fscore", "Precision", "Recall", "AR", "Entropy"};

    for (int dsi = 1; dsi < ds.size(); dsi++) {
        vector<int> answer = {};
        // load data & make folder
        string dataName = ds[dsi];
        cout << dataName << endl;
        string path = dsPath + dataName;
    }
}