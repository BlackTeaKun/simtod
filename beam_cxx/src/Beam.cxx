#include "Beam.hpp"

Beam::Beam(const std::vector<double> &para){
    g = para[0];
    x = para[1];
    y = para[2];
    s = para[3];
    p = para[4];
    c = para[5];

    double s_square = s*s;
    Eigen::Matrix2d cov;
    cov << (1+p), c,
           c    , (1-p);
    cov *= s_square;
    inv_cov = cov.inverse();
}


double Beam::operator()(double _x, double _y) const{
    Eigen::Vector2d vec_x(_x-x, _y-y);
    double expo = -0.5 * vec_x.transpose()*inv_cov*vec_x;
    return exp(expo);
}
Eigen::MatrixXd Beam::operator()(const Eigen::MatrixXd &_xy_coord) const{
    Eigen::Vector2d _center(x,y);
    Eigen::MatrixXd _delta_xy = _xy_coord.colwise() - _center;
    Eigen::RowVectorXd expo_part = (_delta_xy.transpose()*inv_cov*_delta_xy).diagonal() * (-.5);
    Eigen::RowVectorXd res       = expo_part.array().exp();
    res /= res.sum() * g;
    return res;
}
