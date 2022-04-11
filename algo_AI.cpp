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

MatrixXd EuDist2(MatrixXd& A, MatrixXd& B) {}

ret function(vector<MatrixXd>& X, MatrixXd& gt, int d, int argc, char* argv[]) {
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
        // mapstd
    }
    // initialize G,F
    MatrixXd F = MatrixXd::Ones(d, k);
    MatrixXd Y = MatrixXd::Zero(n, 1);
    for (int i = 0; i < k; i++) {
        Y(i, 0) = i;
    }

    MatrixXd alpha = MatrixXd::Ones(1, m) / m;
    // opt.disp = 0
    int flag = 1;  // judge if convergence
    int iter = 0;

    while (flag) {
        iter = iter + 1;
        vector<MatrixXd> XvYT;
        // optimize G_i dv*d
        int myid, numprocs;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        printf("Process %d of %d.\n", myid, numprocs);
        for (int iv = myid; iv < m; iv += numprocs) {
            MatrixXd XvYTtmp = MatrixXd::Zero(X[iv].rows(), k);  // XvYT{iv} computes Xv*Y^T
            XvYT.push_back(XvYTtmp);
            for (int j = 0; j < k; j++) {
                // ...
            }
            Eigen::JacobiSVD<MatrixXd> svd(XvYT[iv] * F.transpose(), Eigen::ComputeFullV | Eigen::ComputeFullU);
            G[iv] = svd.matrixU() * svd.matrixV().transpose();  // svds?
        }
        MPI_Finalize();

        // optimize F d*k
        MatrixXd partJmat(1, 1, 0);
        double J = 0, partJ = 0;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        printf("Process %d of %d.\n", myid, numprocs);
        for (int ia = myid; ia < m; ia += numprocs) {
            partJmat = partJmat + alpha[ia] * G[ia].transpose() * XvYT[ia];
        }
        partJ = partJmat(0, 0);
        MPI_Reduce(&J, &partJ, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        // svds(J)?

        // optimize Y n*1
        MatrixXd loss = MatrixXd::Zero(n, k);
    }
}
