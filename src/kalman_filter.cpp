#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;	// initial state (gaussian mean)
  P_ = P_in;	// initial covariance (gaussian covariance)
  F_ = F_in;	// state transition matrix
  H_ = H_in;	// measurement matrix (projects the state space, e.g. 4D (x,y,x_dot,y_dot) , into the measurement space, e.g. 2D (x,y) )
  R_ = R_in;	// measurement covariance (represents the uncertainty in our sensor measurements; matrix initialized with a fixed value specified by the manufacturer)
  Q_ = Q_in;	// process covariance
}

void KalmanFilter::Predict() {
  /**
  DONE:
    * Prediction
  */
  MatrixXd Ft = F_.transpose();
  // x_ = F_ * x_ + u;  // u = uncertainty, whisch is ignored because x_ is gaussian mean where covairiance P_ represents the uncertainty of the prediction)
  x_ = F_ * x_;
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z, MatrixXd R, MatrixXd H) {
  /**
  DONE:
    * Measurement update
  */
  VectorXd y = z - H * x_;        // (2,1) - (2,4) * (4,1)  ==> (2,1)
  MatrixXd Ht = H_.transpose();   // (2,4)  ==> (4,2)
  MatrixXd S = H_ * P_ * Ht + R;  // (2,4) * (4,4) * (4,2) + (2,2) ==> (2,2)
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;     // (4,4) * (4,2) * (2,2)  ==> (4,2)

  // new state
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  x_ = x_ + (K * y);              // (4,1) + (4,2) * (2,1)  ==> (4,1)
  P_ = (I - K * H) * P_;          // ((4,4) - (4,2) * (2,4)) * (4,4)  ==> (4,4)
}

void KalmanFilter::UpdateEKF(const VectorXd &z, MatrixXd R, MatrixXd Hj) {
  /**
  DONE:
    * update the state by using Extended Kalman Filter equations
  */
  VectorXd y = z - Hj * x_;         // (3,1) - (3,4) * (4,1)  ==> (3,1)
  MatrixXd Hjt = Hj.transpose();    // (3,4)  ==> (4,3)
  MatrixXd S = Hj * P_ * Hjt + R;   // (3,4) * (4,4) * (4,3) + (3,3)  ==> (3,3)
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Hjt * Si;      // (4,4) * (4,3) * (3,3)  ==> (4,3)

  // new state
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  x_ = x_ + (K * y);                // (4,1) + (4,3) * (3,1)  ==> (4,1)
  P_ = (I - K * Hj) * P_;           // ((4,4) - (4,3) * (3,4)) * (4,4)  ==> (4,4)
}
