#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

#define ZERO (0.0001F)
#define PI (3.14159265F)

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
    * Measurement update function
  */
  MatrixXd P_Ht = P_ * H.transpose();  // do this calculation once in advance to eliminate executing twice below.
  VectorXd y = z - H * x_;     // (2,1) - (2,4) * (4,1)  ==> (2,1)
  MatrixXd S = H * P_Ht + R;   // (2,4) * (4,4) * (4,2) + (2,2) ==> (2,2)
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_Ht * Si;     // (4,4) * (4,2) * (2,2)  ==> (4,2)

  // New state
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  x_ = x_ + (K * y);              // (4,1) + (4,2) * (2,1)  ==> (4,1)
  P_ = (I - K * H) * P_;          // ((4,4) - (4,2) * (2,4)) * (4,4)  ==> (4,4)
}

void KalmanFilter::UpdateEKF(const VectorXd &z, MatrixXd R, MatrixXd H) {
  /**
    * Measurement update function using extended kalman filter equations
  */
  // Calculate h(x) vector for EKF measurement update
  VectorXd h(3);
  double px = x_[0];
  double py = x_[1];
  double p_sq = px*px + py*py;
  if(fabs(p_sq) < ZERO) {
    h << 0.0, 0.0, 0.0;  // h(t) approximate to all zeros when current state px,py are nearly zero.
  }
  else {
    double vx = x_[2];
    double vy = x_[3];
    double p = sqrt(p_sq);
    double h1 = p;
    double h2 = atan2(py, px);
    double h3 = (px*vx + py*vy)/p;
    h << h1, h2, h3;
  }

  // Do EKF measurement update
  MatrixXd P_Ht = P_ * H.transpose();  // do this calculation once in advance to eliminate executing twice below.
  VectorXd y = z - h;          // (3,1) - (3,1)  ==> (3,1)
  MatrixXd S = H * P_Ht + R;   // (3,4) * (4,4) * (4,3) + (3,3)  ==> (3,3)
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_Ht * Si;     // (4,4) * (4,3) * (3,3)  ==> (4,3)

  // Normalize radar theta measurement to range <-pi:pi>
  if (y[1] < -PI)
	  y[1] += 2*PI;
  if (y[1] > PI)
	  y[1] -= 2*PI;

  // New state
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  x_ = x_ + (K * y);                // (4,1) + (4,3) * (3,1)  ==> (4,1)
  P_ = (I - K * H) * P_;           // ((4,4) - (4,3) * (3,4)) * (4,4)  ==> (4,4)
}
