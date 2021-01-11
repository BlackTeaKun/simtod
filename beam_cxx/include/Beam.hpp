#pragma once
#include <vector>
#include "Eigen/Dense"
#include <functional>

class Beam{
public:
  Beam(const std::vector<double> &, bool=true);

  double operator()(double, double) const;
  Eigen::MatrixXd operator()(const Eigen::MatrixXd &) const;


  double g, x, y, s, p, c;

private:
  double step;
  Eigen::Matrix2d inv_cov;
  Eigen::MatrixXd cache_data;
  Eigen::MatrixXd(Beam:: *_pointer_to_call) (const Eigen::MatrixXd &) const;

  Eigen::MatrixXd _helper_analytical(const Eigen::MatrixXd &) const;
  Eigen::MatrixXd _helper_numerical(const Eigen::MatrixXd &) const;
  int test(int, int){return 0;}
};
