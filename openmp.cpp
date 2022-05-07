#include <omp.h>

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
using Eigen::MatrixXd;

MatrixXd mapstd(MatrixXd m) {
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
    MatrixXd ans = m;
    return ans;
}

MatrixXd EuDist2(MatrixXd A, MatrixXd B) {
    MatrixXd res(A.rows(), B.rows());
    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < B.rows(); j++) {
            res(i, j) = (A.row(i) - B.row(j)).squaredNorm();
        }
    }
    return res;
}

int main(int argc, char* argv[]) {
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
            MatrixXd Xicopy = Xi;
            Xi = Xicopy.transpose();
            Xi = mapstd(Xi.transpose());
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
        MatrixXd gt = Y;

        // there is no function 'unique' in cpp, so give the k directly
        int k = 10;

        // set the output path
        // ...

        // para setting
        int d = k;  //[k 2*k 3*k]

        for (; d <= 300; d += 5) {
            cout << "d = " << d << endl;
            // tic
            // algo_AI
            // initialize
            int maxIter = 50;
            int k = 10;
            int m = X.size();
            int n = gt.rows();

            vector<MatrixXd> G;
            for (int i = 0; i < m; i++) {
                int di = X[i].cols();
                MatrixXd tmp = MatrixXd::Zero(di, d);
                G.push_back(tmp);
            }
            ofstream outX("./X.txt");
            outX << X[0] << endl;

            // initialize G,F
            MatrixXd F = MatrixXd::Ones(d, k);
            MatrixXd Y = MatrixXd::Zero(n, 1);
            for (int i = 0; i < n; i++) {
                Y(i, 0) = -1;
            }
            for (int i = 0; i < k; i++) {
                Y(i, 0) = i;
            }

            MatrixXd alpha = MatrixXd::Ones(1, m) / m;
            // opt.disp = 0
            int flag = 1;  // judge if convergence
            int iter = 0;
            vector<double> obj;

            while (flag) {
                iter = iter + 1;
                cout << "iter = " << iter << endl;
                vector<MatrixXd> XvYT(m);

                // optimize G_i dv*d
#pragma omp for schedule(dynamic)
                for (int iv = 0; iv < m; iv++) {
                    XvYT[iv] = MatrixXd::Zero(X[iv].rows(), k);  // XvYT{iv} computes Xv*Y^T
                    for (int j = 0; j < k; j++) {
                        for (int a = 0; a < Y.rows(); a++) {
                            if (Y(a, 0) == j) {
                                XvYT[iv].col(j) += X[iv].col(a);
                            }
                        }
                    }
                    Eigen::JacobiSVD<MatrixXd> svd(XvYT[iv] * F.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
                    G[iv] = svd.matrixU() * svd.matrixV().transpose();  // svds?
                }

                ofstream outXv("./XvYT.txt");
                outXv << XvYT[0] * F.transpose() << endl;

                ofstream outG("./G.txt");
                outG << G[0] << endl;

                // optimize F d*k
                vector<MatrixXd> partJ(m);
#pragma omp for schedule(dynamic)
                for (int ia = 0; ia < m; ia++) {
                    partJ[ia] = alpha(0, ia) * G[ia].transpose() * XvYT[ia];
                }
                MatrixXd J = MatrixXd::Zero(partJ[0].rows(), partJ[0].cols());
                for (int i = 0; i < m; i++) {
                    J += partJ[i];
                }
                Eigen::JacobiSVD<MatrixXd> svd_J(J, Eigen::ComputeThinV | Eigen::ComputeThinU);
                F = svd_J.matrixU() * svd_J.matrixV().transpose();

                ofstream outF("./F.txt");
                outF << F << endl;

                // optimize Y n*1
                MatrixXd loss = MatrixXd::Zero(n, k);
                vector<MatrixXd> partloss(m);
#pragma omp for schedule(dynamic)
                for (int ij = 0; ij < m; ij++) {
                    partloss[ij] = alpha(0, ij) * EuDist2(X[ij].transpose(), F.transpose() * G[ij].transpose());
                }
                for (int i = 0; i < m; i++) {
                    loss += partloss[i];
                }
                // min
                for (int i = 0; i < loss.rows(); i++) {
                    MatrixXd::Index minRow, minCol;
                    double min = loss.row(i).minCoeff(&minRow, &minCol);
                    Y(i, 0) = minCol;
                }
                ofstream outY("./Y.txt");
                outY << Y << endl;

                // optimize alpha
                MatrixXd aloss = MatrixXd::Zero(1, m);
                MatrixXd Ftmp = MatrixXd::Zero(F.rows(), Y.rows());
                for (int i = 0; i < Y.rows(); i++) {
                    Ftmp.col(i) = F.col(int(Y(i, 0)));
                }
#pragma omp for schedule(dynamic)
                for (int iv = 0; iv < m; iv++) {
                    aloss(0, iv) = std::sqrt((X[iv] - G[iv] * Ftmp).array().square().sum());
                }
                alpha = 1 / (2 * aloss.array());
                ofstream outa("./alpha.txt");
                outG << alpha << endl;

                //
                MatrixXd term = MatrixXd::Zero(m, 1);
#pragma omp for schedule(dynamic)
                for (int iv = 0; iv < m; iv++) {
                    term(iv, 0) = (X[iv] - G[iv] * Ftmp).array().square().sum();
                }
                obj.push_back((alpha * term)(0, 0));
                ofstream outt("./term.txt");
                outG << term << endl;

                if ((iter > 2) && (abs((obj[iter] - obj[iter - 1]) / (obj[iter])) < 1e-5 || iter > maxIter || obj[iter] < 1e-10)) {
                    flag = 0;
                }
            }

            // toc

            string file_name = "/mnt/d/Code-for-AIMC-master/result/" + to_string(d) + ".txt";
            ofstream out(file_name);
            out << Y + MatrixXd::Constant(Y.rows(), Y.cols(), 1) << endl;
        }
    }
}