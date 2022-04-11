#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mpi.h"
using namespace std;
using Eigen::MatrixXd;

MatrixXd EuDist2(MatrixXd A, MatrixXd B) {
    MatrixXd res(A.rows(), B.rows());
    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < B.rows(); j++) {
            res(i, j) = (A.row(i) - B.row(j)).squaredNorm();
        }
    }
    return res;
}

int main(int argc, char *argv[]) {
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
                // mapstd
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
                    Eigen::JacobiSVD<MatrixXd> svd(XvYT[iv] * F.transpose(), Eigen::ComputeFullV | Eigen::ComputeFullU);
                    G[iv] = svd.matrixU() * svd.matrixV().transpose();  // svds?
                }
                MPI_Finalize();

                // optimize F d*k
                vector<MatrixXd> partJ(m);
                MPI_Init(&argc, &argv);
                MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
                MPI_Comm_rank(MPI_COMM_WORLD, &myid);
                printf("Process %d of %d.\n", myid, numprocs);
                for (int ia = myid; ia < m; ia += numprocs) {
                    partJ[ia] = alpha[ia] * G[ia].transpose() * XvYT[ia];
                }
                MPI_Finalize();
                MatrixXd J = MatrixXd::Zero();
                for (int i = 0; i < m; i++) {
                    J += partJ[i];
                }
                Eigen::JacobiSVD<MatrixXd> svd_J(J, Eigen::ComputeFullV | Eigen::ComputeFullU);
                F = svd_J.matrixU() * svd_J.matrixV().transpose();

                // optimize Y n*1
                vector<MatrixXd> partloss(m);
                MPI_Init(&argc, &argv);
                MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
                MPI_Comm_rank(MPI_COMM_WORLD, &myid);
                printf("Process %d of %d.\n", myid, numprocs);
                for (int ij = myid; ij < m; ij += numprocs) {
                    partloss[ij] = alpha[ij] * EuDist2(X[ij].transpose(), F.transpose() * G[ij].transpose());
                }
                MPI_Finalize();
                MatrixXd loss = MatrixXd::Zero();
                for (int i = 0; i < m; i++) {
                    loss += partloss[i];
                }
                // min?

                // optimize alpha
                MatrixXd aloss = MatrixXd::Zero(1, m);
                MPI_Init(&argc, &argv);
                MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
                MPI_Comm_rank(MPI_COMM_WORLD, &myid);
                printf("Process %d of %d.\n", myid, numprocs);
                for (int iv = myid; iv < m; iv += numprocs) {
                    // sum
                }
                MPI_Finalize();
                alpha = 1 / (2 * aloss.array());

                //
                MatrixXd term = MatrixXd::Zero(m, 1);
                MPI_Init(&argc, &argv);
                MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
                MPI_Comm_rank(MPI_COMM_WORLD, &myid);
                printf("Process %d of %d.\n", myid, numprocs);
                for (int iv = myid; iv < m; iv += numprocs) {
                    // sum
                }
                MPI_Finalize();
                obj.push_back((alpha * term)(0, 0));

                if ((iter > 2) && (abs((obj[iter] - obj[iter - 1]) / (obj[iter])) < 1e-5 || iter > maxIter || obj[iter] < 1e-10)) {
                    flag = 0;
                }
            }

            // toc

            // res = ...
        }
    }
}