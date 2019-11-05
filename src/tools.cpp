#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if(estimations.size() != ground_truth.size() || estimations.size() == 0)
  {
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  for(unsigned int i=0; i < estimations.size(); ++i)
  {
    VectorXd residual = estimations[i] - ground_truth[i];
    //coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse    += residual;
  }

  //calculate the mean
  rmse = rmse / estimations.size();
  //calculate the squared root
  rmse = rmse.array().sqrt();
  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  
  MatrixXd Hj(3,4);
  //recover state parameters
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  /*if (fabs(px) < 0.00001 && fabs(py) < 0.00001){
    px = 0.00001;
    py = 0.00001;
  }*/

  //pre-compute a set of terms to avoid repeated calculation
  double c1 = px*px+py*py;
  //check division by zero
  if(fabs(c1) < 0.000001)
  {
    c1 = 0.000001;
  }

  double c2 = sqrt(c1);
  double c3 = (c1*c2);
  double c4 = py*(vx*py - vy*px);
  double c5 = px*(px*vy - py*vx);

  //compute the Jacobian matrix
  Hj << (px/c2),    (py/c2),        0,    0,
        -(py/c1),   (px/c1),        0,     0,
        c4/c3, 		c5/c3, 		px/c2, py/c2;

  return Hj;
}
