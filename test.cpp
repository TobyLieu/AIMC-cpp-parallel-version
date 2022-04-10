#include <Eigen/Dense>
#include <iostream>
using namespace std;
using Eigen::MatrixXd;
int main() {
    MatrixXd m(2, 2, 0);  // MatrixXd表示是任意尺寸的矩阵ixj, m(2,2)代表一个2x2的方块矩阵
    cout << m << endl;    //输出矩阵m
}