#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mpi.h"
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
        vector<int> d = {3 * k};  //[k 2*k 3*k]

        for (int id = 0; id < d.size(); id++) {
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
                MatrixXd tmp = MatrixXd::Zero(di, d[id]);
                G.push_back(tmp);
                mapstd(X[i]);  // mapstd X已经转置过了，不用再转置
            }
            // initialize G,F
            MatrixXd F = MatrixXd::Ones(d[id], k);
            MatrixXd Y = MatrixXd::Zero(n, 1);
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

                vector<MatrixXd> XvYT(m);
                vector<MatrixXd> partJ(m);
                // MatrixXd J = MatrixXd::Zero(d[id], k);
                MatrixXd J;
                vector<MatrixXd> partloss(m);
                MatrixXd loss;
                MatrixXd aloss;
                MatrixXd Ftmp;
                MatrixXd term = MatrixXd::Zero(m, 1);

                // optimize G_i dv*d
                int myid, numprocs;
                MPI_Init(&argc, &argv);
                MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
                MPI_Comm_rank(MPI_COMM_WORLD, &myid);
                printf("Process %d of %d.\n", myid, numprocs);
                for (int iv = myid; iv < m; iv += numprocs) {
                    XvYT[iv] = MatrixXd::Zero(X[iv].rows(), k);  // XvYT{iv} computes Xv*Y^T
                    for (int j = 0; j < k; j++) {
                        for (int a = 0; a < X[iv].cols(); a++) {
                            if (Y(a, 0) == j) {
                                XvYT[iv].col(j) += X[iv].col(a);
                            }
                        }
                    }
                    Eigen::JacobiSVD<MatrixXd> svd(XvYT[iv] * F.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
                    G[iv] = svd.matrixU() * svd.matrixV().transpose();  // svds?
                }

                MPI_Barrier(MPI_COMM_WORLD);

                // optimize F d*k
                for (int ia = myid; ia < m; ia += numprocs) {
                    partJ[ia] = alpha(0, ia) * G[ia].transpose() * XvYT[ia];
                }
                MPI_Barrier(MPI_COMM_WORLD);
                if (myid == 0) {
                    J = MatrixXd::Zero(partJ[0].rows(), partJ[0].cols());
                    for (int i = 0; i < m; i++) {
                        J += partJ[i];
                    }
                    Eigen::JacobiSVD<MatrixXd> svd_J(J, Eigen::ComputeThinV | Eigen::ComputeThinU);
                    F = svd_J.matrixU() * svd_J.matrixV().transpose();
                }

                MPI_Barrier(MPI_COMM_WORLD);

                // optimize Y n*1
                for (int ij = myid; ij < m; ij += numprocs) {
                    partloss[ij] = alpha(0, ij) * EuDist2(X[ij].transpose(), F.transpose() * G[ij].transpose());
                }
                MPI_Barrier(MPI_COMM_WORLD);
                if (myid == 0) {
                    for (int i = 0; i < m; i++) {
                        loss += partloss[i];
                    }
                    // min
                    for (int i = 0; i < loss.rows(); i++) {
                        MatrixXd::Index minRow, minCol;
                        double min = loss.minCoeff(&minRow, &minCol);
                        Y(i, 0) = minCol;
                    }
                }

                MPI_Barrier(MPI_COMM_WORLD);

                // optimize alpha
                if (myid == 0) {
                    aloss = MatrixXd::Zero(1, m);
                    Ftmp = MatrixXd::Zero(Y.rows(), 1);
                    for (int i = 0; i < Y.rows(); i++) {
                        Ftmp(i, 0) = F(i, Y(i, 0));
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
                for (int iv = myid; iv < m; iv += numprocs) {
                    aloss(0, iv) = std::sqrt((X[iv] - G[iv] * Ftmp).array().square().sum());
                }
                MPI_Barrier(MPI_COMM_WORLD);
                if (myid == 0) {
                    alpha = 1 / (2 * aloss.array());
                }

                //
                MPI_Init(&argc, &argv);
                MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
                MPI_Comm_rank(MPI_COMM_WORLD, &myid);
                printf("Process %d of %d.\n", myid, numprocs);
                for (int iv = myid; iv < m; iv += numprocs) {
                    term(iv, 0) = (X[iv] - G[iv] * Ftmp).array().square().sum();
                }
                MPI_Finalize();
                obj.push_back((alpha * term)(0, 0));

                if ((iter > 2) && (abs((obj[iter] - obj[iter - 1]) / (obj[iter])) < 1e-5 || iter > maxIter || obj[iter] < 1e-10)) {
                    flag = 0;
                }
            }

            // toc

            string file_name = "./result.txt";
            ofstream out(file_name);
            out << Y << endl;
        }
    }
}