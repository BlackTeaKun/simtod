#pragma once
#include <vector>
#include "Eigen/Dense"

class Beam{
public:
  Beam(const std::vector<double> &);

  double operator()(double, double) const;
  Eigen::MatrixXd operator()(const Eigen::MatrixXd &) const;


  double g, x, y, s, p, c;

private:
  Eigen::Matrix2d inv_cov;
  Eigen::MatrixXd cache_data;
};
