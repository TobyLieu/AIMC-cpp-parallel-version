#include <Eigen/Dense>
#include <iostream>
using namespace std;
using Eigen::MatrixXd;

MatrixXd EuDist2(MatrixXd& A, MatrixXd& B) {
    MatrixXd res(A.rows(), B.rows());
    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < B.rows(); j++) {
            res(i, j) = (A.row(i) - B.row(j)).squaredNorm();
            cout << A.row(i) << endl;
            cout << B.row(j) << endl;
            cout << A.row(i) - B.row(j) << endl;
        }
    }
    return res;
}
int main() {
    MatrixXd m(2, 2);
    m(0, 0) = 1;
    m(1, 0) = 3;
    m(0, 1) = 2;
    m(1, 1) = 4;
    MatrixXd n(2, 2);
    n(0, 0) = 4;
    n(1, 0) = 2;
    n(0, 1) = 3;
    n(1, 1) = 1;
    cout << EuDist2(m, n) << endl;
}