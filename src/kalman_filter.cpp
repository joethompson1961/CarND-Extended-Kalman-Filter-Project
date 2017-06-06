#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

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
    * Prediction
  */
  MatrixXd Ft = F_.transpose();
  // x_ = F_ * x_ + u;  // uncertainty 'u' is ignored because it's a gaussian with mean = 0; covairiance P_ represents the uncertainty of the prediction.
  x_ = F_ * x_;
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z, MatrixXd R, MatrixXd H) {
  /**
    * Measurement update
  */
  VectorXd y = z - H * x_;        // (2,1) - (2,4) * (4,1)  ==> (2,1)
  MatrixXd Ht = H.transpose();    // (2,4)  ==> (4,2)
  MatrixXd S = H * P_ * Ht + R;   // (2,4) * (4,4) * (4,2) + (2,2) ==> (2,2)
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;     // (4,4) * (4,2) * (2,2)  ==> (4,2)

  // new state
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  x_ = x_ + (K * y);              // (4,1) + (4,2) * (2,1)  ==> (4,1)
  P_ = (I - K * H) * P_;          // ((4,4) - (4,2) * (2,4)) * (4,4)  ==> (4,4)
}

void KalmanFilter::UpdateEKF(const VectorXd &z, MatrixXd R, MatrixXd H) {
  /**
    * Measurement update using Extended Kalman Filter equations
  */
  float pi = 3.14159265;
  double px = x_[0];
  double py = x_[1];
  double vx = x_[2];
  double vy = x_[3];
  double p_sq;
  double p;
  double h1 = 0.0;
  double h2 = 0.0;
  double h3 = 0.0;
  VectorXd h(3);

  p_sq = px*px + py*py;
  p = sqrt(p_sq);
  h1 = p;
  h2 = atan2(py, px);
  if(fabs(p_sq) < 0.0001)
    h3 = 0;		// Avoid divide by zero error; h3 should approximate to "0" in this case.
  else
    h3 = (px*vx + py*vy)/p;
  h << h1, h2, h3;

  VectorXd y = z - h;             // (3,1) - (3,1)  ==> (3,1)
  MatrixXd Ht = H.transpose();    // (3,4)  ==> (4,3)
  MatrixXd S = H * P_ * Ht + R;   // (3,4) * (4,4) * (4,3) + (3,3)  ==> (3,3)
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;      // (4,4) * (4,3) * (3,3)  ==> (4,3)

  // New state.  First ensure theta (y[1]) is constrained to range -pi:pi (in radians).
  if (y[1] < -pi)
	  y[1] += 2*pi;
  if (y[1] > pi)
	  y[1] -= 2*pi;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  x_ = x_ + (K * y);                // (4,1) + (4,3) * (3,1)  ==> (4,1)
  P_ = (I - K * H) * P_;           // ((4,4) - (4,3) * (3,4)) * (4,4)  ==> (4,4)
}
