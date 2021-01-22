#pragma once

//#include "healpix_base.h"
#include "Eigen/Dense"
#include "Eigen/StdVector"
#include <vector>

template<class _type>
class T_Healpix_Base;

struct Scan_data{
  Scan_data(int _nsample, double *_tod1, double *_tod2, 
      double *_theta, double *_phi, double *_psi ){
    nsample = _nsample;
    tod1    = _tod1 ;
    tod2    = _tod2 ;
    theta   = _theta;
    phi     = _phi  ;
    psi     = _psi  ;
    if((tod1 == nullptr) && (tod2 == nullptr)){
      scanonly = true;
    }
    else{
      scanonly = false;
    }
  }
  double *tod1, *tod2;
  double *theta, *phi, *psi;
  int nsample;
  bool scanonly;
};

class MapMaking{
public:
  using RowMajorDM = 
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  MapMaking(int _nside, bool = false);
  ~MapMaking();
  int add_Scan(const Scan_data &);
  RowMajorDM         get_map();
  Eigen::RowVectorXi get_hitmap();
  bool is_complex;
private:
  int nside, npix;
  Eigen::RowVectorXi hitmap;
  Eigen::MatrixXd    Tmap;

  //least square b=(X@X.T)^{-1}@X.T@y
  std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> x_xt;
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> xt_y;
  T_Healpix_Base<int> *hb;
};
