#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mpi.h"
using namespace std;
using Eigen::MatrixXd;

struct ret {
    int a;
};

ret function(vector<MatrixXd>& X, MatrixXd& gt, int d) {
    // initialize
    int maxIter = 50;
    int k = 10;
    int m = X.size();
    int n = gt.rows();
}
