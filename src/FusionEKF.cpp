#include "FusionEKF.h"
#include "Eigen/Dense"
#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

#define ZERO (0.001F)

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // measurement covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  ekf_.H_ = MatrixXd(2, 4);
  ekf_.H_ << 1, 0, 0, 0,
             0, 1, 0, 0;

  // create initial state transition matrix F_
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

  /**
    * Set the process and measurement noises
  */
  // create initial state covariance matrix P_
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;

  // acceleration noise components (process noise)
  ekf_.noise_ax_ = 9;
  ekf_.noise_ay_ = 9;
  ekf_.Q_ = MatrixXd(4, 4);
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
    */
    ekf_.x_ = VectorXd(4);
//    ekf_.x_ << 1, 1, 5, 0;  // Note: default velocities vx=5, vy=0 are based on observation of dataset 1 and are probably wrong for dataset 2.
    ekf_.x_ << 1, 1, 0, 0;  // Note: default velocities vx=5, vy=0 are based on observation of dataset 1 and are probably wrong for dataset 2.

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
        * Initialize state vector with location.
        * For radar, convert from polar to cartesian coordinates
      */
      float ro = measurement_pack.raw_measurements_[0];
  	  float theta = measurement_pack.raw_measurements_[1];
  	  float ro_dot = measurement_pack.raw_measurements_[2];
      ekf_.x_[0] = ro * cos(theta);
      ekf_.x_[1] = ro * sin(theta);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
        * Initialize state vector with location.
      */
      ekf_.x_[0] = measurement_pack.raw_measurements_[0];
      ekf_.x_[1] = measurement_pack.raw_measurements_[1];
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  double delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	// delta time is expressed in seconds
  // Only do a prediction update if more than zero time has passed since last measurement
  if (delta_t > ZERO) {
    previous_timestamp_ = measurement_pack.timestamp_;
    ekf_.Predict(delta_t);
  }
  else
    cout << "Zero time measurement update!";

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  /**
     * Based on the sensor type, perform the appropriate measurement update step
     * and update the state and covariance matrices.
   */
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Eigen::VectorXd z = VectorXd(3);
	z << measurement_pack.raw_measurements_;

	ekf_.UpdateEKF(z, R_radar_);
//    // print the output
//    cout << "RADAR  z:" << endl << z << endl;
//    cout << "x_ = " << endl << ekf_.x_ << endl;
//    cout << "P_ = " << endl << ekf_.P_ << endl;
  } else {
    // Laser updates
    Eigen::VectorXd z = VectorXd(2);
	z << measurement_pack.raw_measurements_;

    ekf_.Update(z, R_laser_);
//    // print the output
//    cout << "LIDAR  z:" << endl << z << endl;
//    cout << "x_ = " << endl << ekf_.x_ << endl;
//    cout << "P_ = " << endl << ekf_.P_ << endl;
  }

}
