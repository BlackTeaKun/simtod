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
    if((tod1 == nullptr) || (tod2 == nullptr) || (psi == nullptr)){
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
  using _eigen_type = 
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using _eigen_int_type = 
    Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  MapMaking(int _nside);
  ~MapMaking();
  int add_Scan(const Scan_data &);
  _eigen_type     get_map();
  _eigen_int_type get_hitmap();
private:
  int nside, npix;
  _eigen_int_type hitmap;
  _eigen_type     Tmap;

  //least square b=(X@X.T)^{-1}@X.T@y
  std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> x_xt;
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> xt_y;
  T_Healpix_Base<int> *hb;
};
