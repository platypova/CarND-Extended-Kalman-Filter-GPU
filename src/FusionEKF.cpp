#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  
  is_initialized_ = false;
  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0,      0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0,      0,
              0,    0.0009, 0,
              0,    0,      0.09;

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  
  Hj_<< 1, 1, 0, 0,
  		1, 1, 0, 0,
  		1, 1, 1, 1;

  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

  //state covariance matrix P
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
			 0, 0, 0, 1;
  
  noise_ax = 9;
  noise_ay = 9;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  //Initialization
  if (!is_initialized_) {
    //Initialize the state ekf_.x_ with the first measurement.
    //Create the covariance matrix.
    // first measurement
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1,1,1,1;
    
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
    {
     
      //Convert radar from polar to cartesian coordinates and initialize state.
      double rho     = measurement_pack.raw_measurements_[0];
      double phi     = measurement_pack.raw_measurements_[1]; 
      double rho_dt = measurement_pack.raw_measurements_[2];

      // normalize angle between -pi and pi:
      phi = atan2(sin(phi),cos(phi));
      // Convert coordinates:
      double rx  = rho * cos(phi);
      double ry  = rho * sin(phi);
      double rvx = rho_dt * cos(phi);
      double rvy = rho_dt * sin(phi);    
          
      ekf_.x_ << rx, ry, rvx, rvy;
      
    } 
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) 
    {
      double lx = measurement_pack.raw_measurements_[0];
      double ly = measurement_pack.raw_measurements_[1];
      ekf_.x_(0) = lx;
      ekf_.x_(1) = ly;
    }
    
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  //Prediction
  //Update the state transition matrix F according to the new elapsed time.
  //Time is measured in seconds.
  //Update the process noise covariance matrix.
   
  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  // State transition matrix:
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;
  
  double dt2   = dt * dt;
  double dt3   = dt2 * dt;
  double dt4   = dt3 * dt;

  // process covariance matrix calculation:
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt4*noise_ax/4, 0,               dt3*noise_ax/2, 0,
             0,               dt4*noise_ay/4, 0,               dt3*noise_ay/2,
             dt3*noise_ax/2, 0,               dt2*noise_ax,   0,
             0,               dt3*noise_ay/2, 0,               dt2*noise_ay;

  ekf_.Predict();

  //Update

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
  {  
    Tools tools;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } 
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) 
  {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}