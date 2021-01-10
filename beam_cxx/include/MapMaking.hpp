#pragma once

#include "healpix_base.h"
#include "Eigen/Dense"

template<class _data_type, class _ptg_type>
struct Scan_data{
  Scan_data(int _nsample, _data_type *_tod1, _data_type *_tod2, 
      _ptg_type *_theta, _ptg_type *_phi, _ptg_type *_psi ){
    nsample = _nsample;
    tod1    = _tod1 ;
    tod2    = _tod2 ;
    theta   = _theta;
    phi     = _phi  ;
    psi     = _psi  ;
  }
  _data_type *tod1, *tod2;
  _ptg_type *theta, *phi, *psi;
  int nsample;
};

template<class _data_type>
struct MapBase{
  using _eigen_type =
    Eigen::Map<Eigen::Matrix<_data_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  MapBase() = delete;
  MapBase(int, bool);
  ~MapBase();
  void free_raw_data();

  int nside;
  bool pol;
  _eigen_type map_data;
private:
  _data_type *raw_data;
};

template<class _data_type>
MapBase<_data_type>::MapBase(int _nside, bool _pol){
  nside=_nside;
  pol = _pol;
  int npix = 12 * nside*nside;
  if(pol)
  {
    raw_data = new _data_type[npix*3];
    map_data = _eigen_type(raw_data, 3, npix);
  }
  else
  {
    raw_data = new _data_type[npix];
    map_data = _eigen_type(raw_data, 1, npix);
  }
  map_data.array() = 0;
}
template<class _data_type>
void MapBase<_data_type>::free_raw_data()
{
  if(raw_data == nullptr)
  {
    return;
  }
  delete[] raw_data;
  raw_data = nullptr;
}


template<class _data_type, class _ptg_type>
class MapMaking{
public:
  MapMaking(int _nside, bool _pol);
  ~MapMaking();
  int add_Scan(const Scan_data<_data_type, _ptg_type> &);
  MapBase<_data_type> *get_map();
  MapBase<int>        *get_hitmap();
private:
  int nside;
  bool pol;
  MapBase<int> *hitmap;
  MapBase<_data_type> *tempmap, *resmap;
  T_Healpix_Base<int> hb;
};
