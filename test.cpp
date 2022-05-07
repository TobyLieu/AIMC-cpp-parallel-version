#include <Eigen/Dense>
#include <iostream>
using namespace std;
using Eigen::MatrixXd;

void mapstd(MatrixXd& m) {
    Eigen::MatrixXd mean = m.rowwise().mean();  //<==> m.rowwise().sum() / m.cols();
    for (int i = 0; i < m.rows(); i++) {
        double mean_ = mean(i, 0);
        double scale = 1. / (m.cols() - 1);
        double sqsum_ = 0;
        for (int j = 0; j < m.cols(); j++) {
            sqsum_ += (m(i, j) - mean_) * (m(i, j) - mean_);
        }
        double variance_ = sqsum_ * scale;
        double stddev_ = std::sqrt(variance_);
        // cout << mean_ << " " << variance_ << " " << stddev_ << endl;
        m.row(i) -= MatrixXd::Constant(1, m.cols(), mean_);
        m.row(i) /= stddev_;
    }
}

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
    MatrixXd m(3, 2);
    m(0, 0) = 1;
    m(0, 1) = 2;
    m(1, 0) = 3;
    m(1, 1) = 4;
    m(2, 0) = 5;
    m(2, 1) = 6;
    cout << m << endl;
    MatrixXd n(2, 2);
    n(0, 0) = 4;
    n(1, 0) = 2;
    n(0, 1) = 3;
    n(1, 1) = 1;
    mapstd(n);
    MatrixXd Y(m.rows(), 1);
    for (int i = 0; i < m.rows(); i++) {
        MatrixXd::Index minRow, minCol;
        double min = m.minCoeff(&minRow, &minCol);
        Y(i, 0) = minCol;
    }
    Eigen::JacobiSVD<MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
    cout << svd.matrixU() * svd.matrixV().transpose() << endl;
}