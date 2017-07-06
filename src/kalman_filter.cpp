#include "kalman_filter.h"
#include <iostream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

#define ZERO (0.0001F)
#define PI (3.14159265F)

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;	// state (gaussian mean), e.g. (x, y, x_dot, y_dot)
  P_ = P_in;	// covariance (gaussian covariance)
  F_ = F_in;	// state transition matrix
  H_ = H_in;	// measurement matrix (projects the state space, e.g. 4D (x,y,x_dot,y_dot) , into the measurement space, e.g. 2D (x,y) )
  R_ = R_in;	// measurement covariance (represents the uncertainty in our sensor measurements; matrix initialized with a fixed value specified by the manufacturer)
  Q_ = Q_in;	// process covariance
}

void KalmanFilter::Predict(double delta_t) {
  /**
    * Prediction
  */
  // update the process noise covariance matrix Q for new elapsed time
  double dt_2 = delta_t * delta_t;
  double dt_3 = dt_2 * delta_t;
  double dt_4 = dt_3 * delta_t;
//  Q_ = MatrixXd(4, 4);
  Q_ <<  dt_4/4*noise_ax_,    0,                dt_3/2*noise_ax_, 0,
         0,                   dt_4/4*noise_ay_, 0,                dt_3/2*noise_ay_,
         dt_3/2*noise_ax_,    0,                dt_2*noise_ax_,   0,
         0,                   dt_3/2*noise_ay_, 0,                dt_2*noise_ay_;

  // update state transition matrix F for new elapsed time
  F_(0, 2) = delta_t;
  F_(1, 3) = delta_t;

  // predict next state x_ and uncertainty P_
  MatrixXd Ft = F_.transpose();
  // x_ = F_ * x_ + u;  // uncertainty 'u' is ignored because it's a gaussian with mean = 0; covairiance P_ represents the uncertainty of the prediction.
  x_ = F_ * x_;               // (4,4) * (4,1)  ==> (4,1)
  P_ = F_ * P_ * Ft + Q_;     // ((4,4) * (4,4) * (4,4)) + (4,4) ==> (4,4)
}

void KalmanFilter::Update(const VectorXd &z, MatrixXd R) {
  /**
    * Measurement update function
  */
  MatrixXd P_Ht = P_ * H_.transpose(); // do this calculation once in advance to eliminate executing twice below.
  VectorXd y = z - H_ * x_;            // difference between measurement and predicted measurement (2,1) - (2,4) * (4,1)  ==> (2,1)
  MatrixXd S = H_ * P_Ht + R;          // innovation covariance (2,4) * (4,4) * (4,2) + (2,2) ==> (2,2)
  MatrixXd K = P_Ht * S.inverse();     // kalman gain (4,4) * (4,2) * (2,2)  ==> (4,2)

  // New state
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  x_ = x_ + (K * y);              // (4,1) + (4,2) * (2,1)  ==> (4,1)
  P_ = (I - K * H_) * P_;         // ((4,4) - (4,2) * (2,4)) * (4,4)  ==> (4,4)
}

void KalmanFilter::UpdateEKF(const VectorXd &z, MatrixXd R) {
  /**
    * Measurement update function using extended kalman filter equations
    * x_: the predicted state
    * z: radar measurement
    * R: radar measurement covariance matrix
    * H: jacobian measurement matrix
    * h: predicted state represented in radar measurement space, i.e. the predicted measurement
    * y: predicted vs actual measurement difference
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
    // convert state vector to radar measurement space - THE NON-LINEAR PART!!
    double vx = x_[2];
    double vy = x_[3];
    double p = sqrt(p_sq);
    double h1 = p;
    double h2 = atan2(py, px);
    double h3 = (px*vx + py*vy)/p;
    h << h1, h2, h3;
  }

  MatrixXd Hj = tools.CalculateJacobian(ekf_.x_);

  // Do EKF measurement update
  MatrixXd P_Ht = P_ * Hj.transpose();  // do this calculation once in advance to eliminate executing twice below.
  VectorXd y = z - h;       // difference between measurement and predicted measurement (3,1) - (3,1)  ==> (3,1)
  MatrixXd S = Hj * P_Ht + R;// innovation covariance (3,4) * (4,4) * (4,3) + (3,3)  ==> (3,3)
  MatrixXd K =  P_Ht * S.inverse();  // kalman gain (4,4) * (4,3) * (3,3)  ==> (4,3)

  // Normalize radar theta measurement to range <-pi:pi>
  while (y[1] < -PI) {y[1] += 2*PI;}
  while (y[1] > PI)  {y[1] -= 2*PI;}

  // New state
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  x_ = x_ + (K * y);                // (4,1) + (4,3) * (3,1)  ==> (4,1)
  P_ = (I - K * Hj) * P_;           // ((4,4) - (4,3) * (3,4)) * (4,4)  ==> (4,4)
}
