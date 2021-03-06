#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include "Eigen/Dense"
#include "tools.h"

class KalmanFilter {
public:

  // state vector
  Eigen::VectorXd x_;

  // state covariance matrix
  // represents the uncertainty of the current state
  Eigen::MatrixXd P_;

  // state transition matrix:
  // used to predict the next state from the current state
  Eigen::MatrixXd F_;

  // process noise covariance matrix:
  // generated prior to each prediction
  // dependent on time passed and hard coded noise coefficients.
  Eigen::MatrixXd Q_;

  // measurement matrix:
  // projects the state space, e.g. 4D {x,y,x_dot,y_dot}, into the measurement space, e.g. 2D (x,y)
  Eigen::MatrixXd H_;

  // measurement covariance matrix:
  // represents the uncertainty in sensor measurements
  // initialized with a fixed value specified by the manufacturer
  Eigen::MatrixXd R_;

  //acceleration noise components
  float noise_ax_;
  float noise_ay_;

  /**
   * Constructor
   */
  KalmanFilter();

  /**
   * Destructor
   */
  virtual ~KalmanFilter();

  /**
   * Init Initializes Kalman filter
   * @param x_in Initial state
   * @param P_in Initial state covariance
   * @param F_in Transition matrix
   * @param H_in Measurement matrix
   * @param R_in Measurement covariance matrix
   * @param Q_in Process covariance matrix
   */
  void Init(Eigen::VectorXd &x_in, Eigen::MatrixXd &P_in, Eigen::MatrixXd &F_in,
      Eigen::MatrixXd &H_in, Eigen::MatrixXd &R_in, Eigen::MatrixXd &Q_in);

  /**
   * Prediction Predicts the state and the state covariance
   * using the process model
   * @param delta_T Time between k and k+1 in s
   */
  void Predict(double delta_t);

  /**
   * Updates the state by using standard Kalman Filter equations
   * @param z The measurement at k+1
   */
  void Update(const Eigen::VectorXd &z, Eigen::MatrixXd R);

  /**
   * Updates the state by using Extended Kalman Filter equations
   * @param z The measurement at k+1
   */
  void UpdateEKF(const Eigen::VectorXd &z, Eigen::MatrixXd R);

private:

  // tool object used to compute Jacobian and RMSE
  Tools tools;
};


#endif /* KALMAN_FILTER_H_ */
