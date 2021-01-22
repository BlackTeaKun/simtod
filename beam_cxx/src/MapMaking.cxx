#include "MapMaking.hpp"
#include "healpix_base.h"
#include "pointing.h"
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>

MapMaking::MapMaking(int _nside, bool _is_complex)
{
    hb = new T_Healpix_Base<int>();
    nside = _nside;
    is_complex = _is_complex;

    hb->SetNside(nside, RING);
    npix = nside * nside *12;

    int ndim = _is_complex ? 2 : 1;
    Tmap     = Eigen::MatrixXd(ndim, npix);
    hitmap   = Eigen::RowVectorXi(npix);
    Tmap.setZero();
    hitmap.setZero();

    x_xt = std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>>(npix);
    xt_y = std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>(npix);
    for(int i=0; i<npix; ++i){
        x_xt[i].setZero();
        xt_y[i].setZero();
    }
}

MapMaking::~MapMaking()
{
  delete hb;
}

int MapMaking::add_Scan(const Scan_data &scan)
{
    int nsample = scan.nsample;
    std::vector<int> idx(nsample);
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < nsample; ++i){
        pointing ptg(scan.theta[i], scan.phi[i]);
        idx[i] = hb->ang2pix(ptg);
    }

    // For Hitmap
    hitmap(Eigen::all, idx).array() += 1;

    if(scan.scanonly){
        return 0;
    }


    // Eigen tod wrapper
    using _tod_type = Eigen::RowVector<double, Eigen::Dynamic>;
    _tod_type pair_sum, pair_diff;
    if(scan.tod2 == nullptr){
        int cur_sample = is_complex ? 2 * nsample : nsample;
        Eigen::Map<_tod_type> tod1(scan.tod1, cur_sample);
        pair_sum  = tod1;
    }
    else{
        Eigen::Map<_tod_type> tod1(scan.tod1, nsample);
        Eigen::Map<_tod_type> tod2(scan.tod2, nsample);
        pair_sum  = (tod1 + tod2) / 2;
        pair_diff = (tod1 - tod2) / 2;
    }


    // For Temperature
    int tmap_rows = Tmap.rows();
    int tmap_cols = Tmap.cols();
    Tmap(Eigen::all, idx) += Eigen::Map<Eigen::MatrixXd>(pair_sum.data(), tmap_rows, pair_sum.size()/tmap_rows);
    //Tmap(Eigen::all, idx).array() += pair_sum.array();
    if(scan.tod2 == nullptr){
        return 0;
    }

    // For Polarization
    using std::cos;
    using std::sin;
    for(int i = 0; i < nsample; ++i)
    {
      double cur_psi = scan.psi[i];
      int    cur_idx = idx[i];

      // For X@X.T
      Eigen::RowVector2d c_s_vector;
      c_s_vector << cos(2*cur_psi), sin(2*cur_psi);

      x_xt[cur_idx].array() += (c_s_vector.transpose()*c_s_vector).array();

      // For X.T@y
      xt_y[cur_idx] += c_s_vector.transpose()*pair_diff[i];
    }
    return 0;
}

typename MapMaking::RowMajorDM MapMaking::get_map()
{
    constexpr double eps = std::numeric_limits<double>::epsilon();
    Eigen::RowVectorXd hitmap_double = hitmap.cast<double>().array() + eps;
    if(is_complex){
        Tmap.array().rowwise() /= hitmap_double.array();
        return Eigen::Map<Eigen::RowVectorXd>(Tmap.data(), 2*npix);
    }
    RowMajorDM result(3, npix);
    result.setZero();

    // For T map
    result.row(0).array() = Tmap.array() / hitmap_double.array();

    // For QU maps
    double tol = 1e-5;
    for(int i=0; i<npix; ++i){
        bool is_invertable = (std::abs(x_xt[i].determinant()) > tol);
        if (!is_invertable) continue;
        result.bottomRows(2).col(i) = x_xt[i].inverse() * xt_y[i];
    }
    return result;
}

Eigen::RowVectorXi MapMaking::get_hitmap()
{
    return hitmap;
}

