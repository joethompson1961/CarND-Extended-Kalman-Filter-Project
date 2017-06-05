#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  DONE:
    * Calculate the RMSE here.
  */
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if(estimations.size() != ground_truth.size()
			|| estimations.size() == 0){
		cout << "Invalid estimation or ground_truth data" << endl;
		return rmse;
	}

	//accumulate squared residuals
	for(unsigned int i=0; i < estimations.size(); ++i){

		VectorXd residual = estimations[i] - ground_truth[i];

		//coefficient-wise multiplication
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	//calculate the mean
	rmse = rmse/estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  DONE:
    * Calculate a Jacobian here.
  */
	MatrixXd Hj(3,4);

	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//DONE: YOUR CODE HERE
    Hj << 0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0;

	float p_sq = px*px + py*py;
	float p = sqrt(p_sq);
	float h1 = py*(vx*py - vy*px);
	float h2 = px*(vy*px - vx*py);

	//check division by zero
	if(fabs(p_sq) < 0.0001){
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		return Hj;
	}

	else
	{
	    //compute the Jacobian matrix
	    Hj << px/p, py/p, 0, 0,
	          -py/p_sq, px/p_sq, 0, 0,
	          h1/(p_sq*p), h2/(p_sq*p), px/p, py/p;
	}

	return Hj;
}
