#include "Beam.hpp"
#include <functional>
#include <iostream>

Beam::Beam(const std::vector<double> &para, bool is_numerical){
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

    _pointer_to_call = is_numerical? &Beam::_helper_numerical :
                                     &Beam::_helper_analytical;

    if(is_numerical)
    {
        int size = 1000;
        cache_data = Eigen::MatrixXd(size,size);
        Eigen::RowVectorXd xy_coord = Eigen::RowVectorXd::LinSpaced(size, -5*s, 5*s);
        step = 10*s / (size - 1);
        for (int xi = 0; xi < size; ++xi)
        {
            for (int yj = 0; yj < size; ++yj)
            {
                cache_data(xi,yj) = this->operator()(xy_coord(xi), xy_coord(yj));
            }
        }
    }
}


double Beam::operator()(double _x, double _y) const{
    Eigen::Vector2d vec_x(_x-x, _y-y);
    double expo = -0.5 * vec_x.transpose()*inv_cov*vec_x;
    return exp(expo);
}
Eigen::MatrixXd Beam::operator()(const Eigen::MatrixXd &_xy_coord) const
{
    return (this->*_pointer_to_call)(_xy_coord);
}

Eigen::MatrixXd Beam::_helper_analytical(const Eigen::MatrixXd &_xy_coord) const{
    Eigen::Vector2d _center(x,y);
    Eigen::MatrixXd    _delta_xy = _xy_coord.colwise() - _center;
    Eigen::RowVectorXd expo_part = (_delta_xy.transpose()*inv_cov*_delta_xy).diagonal() * (-.5);
    Eigen::RowVectorXd res       = expo_part.array().exp();
    double tot = res.sum();
    res *= g/tot;
    return res;
}

Eigen::MatrixXd Beam::_helper_numerical(const Eigen::MatrixXd &_xy_coord) const{
    Eigen::RowVectorXd x = _xy_coord.row(0);
    Eigen::RowVectorXd y = _xy_coord.row(1);

    double xy_init = -5*s;
    Eigen::RowVectorXi x_idx = ((x.array() - xy_init)/step + 0.5).cast<int>();
    Eigen::RowVectorXi y_idx = ((y.array() - xy_init)/step + 0.5).cast<int>();
    Eigen::RowVectorXd res   = cache_data(x_idx, y_idx).diagonal();
    double tot = res.sum();
    res *= g/tot;
    return res;
}
