#include "Beam.hpp"
#include "healpix_base.h"
#include "Eigen/Dense"
#include "convolve.hpp"

#include <iostream>
#include <chrono>

inline std::pair<vec3, vec3> get_tangent_vec(const vec3 &_pt_vec){
  vec3 x_hat(-_pt_vec.y, _pt_vec.x, 0);
  x_hat.Normalize();
  vec3 y_hat = crossprod(_pt_vec, x_hat);
  return std::make_pair(x_hat, y_hat);
}

//Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
double *convolve(const Beam &b1, const Beam &b2, size_t npix, double *p_map, size_t nsample,
      double *p_theta, double *p_phi, double *p_psi){
  double max_radius = std::max(b1.s, b2.s)*4;
  Eigen::Map<Eigen::RowVectorXd> map_wrapper(p_map, npix);

  T_Healpix_Base<int> hb;
  int nside = hb.npix2nside(npix);
  hb.SetNside(nside, RING);


  // raw data of tod;
  double *read_out = new double[nsample * 2];

  // tod data wrapper
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map_tod(read_out, 2, nsample);

#pragma omp parallel for schedule(dynamic, 1)
  for(size_t i = 0; i < nsample; ++i){
#ifdef DEBUG
    if(i%10000 == 0) printf("!----%08lu\n", i);
#endif
    pointing cur_pt(p_theta[i], p_phi[i]);
    auto cur_pt_vec3 = cur_pt.to_vec3();
    auto xy_hat = get_tangent_vec(cur_pt_vec3);
    Eigen::Matrix2d rot_mat_2d;
    rot_mat_2d = Eigen::Rotation2Dd(-p_psi[i]);

    rangeset<int> rs;
    std::vector<int> queried_pix;
    hb.query_disc(cur_pt, max_radius, rs);
    rs.toVector(queried_pix);

    int queried_num = queried_pix.size();
    Eigen::MatrixXd all_fixed_coord(2, queried_num);
    for (int j = 0; j < queried_num; ++j){
      auto queried_j_vec3 = hb.pix2vec(queried_pix[j]);
      auto diff_vec3 = queried_j_vec3 - cur_pt_vec3;
      double x_coord = dotprod(xy_hat.first, diff_vec3);
      double y_coord = dotprod(xy_hat.second, diff_vec3);
      Eigen::Vector2d cur_fix_coord_xy;
      cur_fix_coord_xy << x_coord, y_coord;
      all_fixed_coord.col(j) = cur_fix_coord_xy;
    }
    // Rotate!
    all_fixed_coord.applyOnTheLeft(rot_mat_2d);
    Eigen::RowVectorXd pixel_coef_1 = b1(all_fixed_coord);
    Eigen::RowVectorXd pixel_coef_2 = b2(all_fixed_coord);

    //map_wrapper(queried_pix) = pixel_coef_1 - pixel_coef_2;


    // for tods
    map_tod(0,i) = (map_wrapper(queried_pix).array() * pixel_coef_1.array()).sum();
    map_tod(1,i) = (map_wrapper(queried_pix).array() * pixel_coef_2.array()).sum();

  }

  // return the raw_data (or wrapper?)
  return read_out;
}


double *template_tod(const Beam &b1, const Beam &b2, const DerivTMaps &maps, size_t nsample,
    double *p_theta, double *p_phi, double *p_psi){

  double g[] = {b1.g, b2.g};
  double x[] = {b1.x, b2.x};
  double y[] = {b1.y, b2.y};
  double s[] = {b1.s, b2.s};
  double p[] = {b1.p, b2.p};
  double c[] = {b1.c, b2.c};
  double s_bar = (b1.s + b2.s)/2;
  Eigen::Vector2d g_vector(b1.g, b2.g);
  Eigen::Vector2d x_vector(b1.x, b2.x);
  Eigen::Vector2d y_vector(b1.y, b2.y);
  Eigen::Vector2d s_vector(b1.s, b2.s);
  Eigen::Vector2d p_vector(b1.p, b2.p);
  Eigen::Vector2d c_vector(b1.c, b2.c);

  T_Healpix_Base<int> hb;
  int nside = hb.npix2nside(maps.npix);
  hb.SetNside(nside, RING);

  std::vector<int> idx(nsample);
#pragma omp parallel for schedule(dynamic, 1)
  for(int i = 0; i < nsample; ++i){
    pointing ptg(p_theta[i], p_phi[i]);
    idx[i] = hb.ang2pix(ptg);
  }

  double *tod = new double[2*nsample];
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map_tod(tod, 2, nsample);
  map_tod.setZero();

  using wrapper_type = Eigen::Map<Eigen::ArrayXd>;

  wrapper_type partial_t(maps.dt_s_tmap, maps.npix);
  wrapper_type partial_p(maps.dp_s_tmap, maps.npix);
  wrapper_type partial_tt(maps.dtt_s_tmap, maps.npix);
  wrapper_type partial_pp(maps.dpp_s_tmap, maps.npix);
  wrapper_type partial_tp(maps.dtp_s_tmap, maps.npix);
  wrapper_type smoothed_map(maps.s_tmap, maps.npix);

  wrapper_type psi_vector(p_psi, nsample);
  Eigen::ArrayXd cospsi  = psi_vector.array().cos();
  Eigen::ArrayXd sinpsi  = psi_vector.array().sin();
  Eigen::ArrayXd cos2psi = (psi_vector * 2).array().cos();
  Eigen::ArrayXd sin2psi = (psi_vector * 2).array().sin();

  // Monopole
  // gain
  map_tod += g_vector * smoothed_map(idx).matrix().transpose();

  // beam width
  Eigen::Vector2d delta_s = s_vector.array() - s_bar;
  map_tod += delta_s * s_bar * (partial_tt(idx) + partial_pp(idx)).matrix().transpose();

  // Dipole
  map_tod += x_vector * ( cospsi*partial_p(idx) - sinpsi*partial_t(idx)).matrix().transpose();
  map_tod += y_vector * (-cospsi*partial_t(idx) - sinpsi*partial_p(idx)).matrix().transpose();

  // Quadrupole
  Eigen::ArrayXd partial_xx = cospsi*cospsi*partial_pp(idx) +
                              sinpsi*sinpsi*partial_tt(idx) -
                              sin2psi*partial_tp(idx);
  Eigen::ArrayXd partial_yy = sinpsi*sinpsi*partial_pp(idx) +
                              cospsi*cospsi*partial_tt(idx) +
                              sin2psi*partial_tp(idx);
  Eigen::ArrayXd partial_xy = 0.5*(
                                sin2psi*(partial_tt(idx)-partial_pp(idx)) -
                                2*cos2psi*partial_tp(idx)
                              );

  Eigen::ArrayXd p_xx_minus_p_yy = partial_xx - partial_yy;

  map_tod += 0.5*p_vector * s_bar*s_bar * p_xx_minus_p_yy.matrix().transpose();
  map_tod +=     c_vector * s_bar*s_bar * partial_xy.matrix().transpose();

  return tod;
}

double *template_tod_interp(const Beam &b1, const Beam &b2, const DerivTMaps &maps, size_t nsample,
    double *p_theta, double *p_phi, double *p_psi){

  double g[] = {b1.g, b2.g};
  double x[] = {b1.x, b2.x};
  double y[] = {b1.y, b2.y};
  double s[] = {b1.s, b2.s};
  double p[] = {b1.p, b2.p};
  double c[] = {b1.c, b2.c};
  double s_bar = (b1.s + b2.s)/2;
  Eigen::Vector2d g_vector(b1.g, b2.g);
  Eigen::Vector2d x_vector(b1.x, b2.x);
  Eigen::Vector2d y_vector(b1.y, b2.y);
  Eigen::Vector2d s_vector(b1.s, b2.s);
  Eigen::Vector2d p_vector(b1.p, b2.p);
  Eigen::Vector2d c_vector(b1.c, b2.c);

  T_Healpix_Base<int> hb;
  int nside = hb.npix2nside(maps.npix);
  hb.SetNside(nside, RING);

  double *tod = new double[2*nsample];
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map_tod(tod, 2, nsample);
  map_tod.setZero();

  using wrapper_type = Eigen::Map<Eigen::ArrayXd>;
  wrapper_type partial_t(maps.dt_s_tmap, maps.npix);
  wrapper_type partial_p(maps.dp_s_tmap, maps.npix);
  wrapper_type partial_tt(maps.dtt_s_tmap, maps.npix);
  wrapper_type partial_pp(maps.dpp_s_tmap, maps.npix);
  wrapper_type partial_tp(maps.dtp_s_tmap, maps.npix);
  wrapper_type smoothed_map(maps.s_tmap, maps.npix);

  wrapper_type psi_vector(p_psi, nsample);
  Eigen::ArrayXd cospsi  = psi_vector.array().cos();
  Eigen::ArrayXd sinpsi  = psi_vector.array().sin();
  Eigen::ArrayXd cos2psi = (psi_vector * 2).array().cos();
  Eigen::ArrayXd sin2psi = (psi_vector * 2).array().sin();

  Eigen::ArrayXd smoothed_map_interp(nsample);
  Eigen::ArrayXd partial_t_interp(nsample);
  Eigen::ArrayXd partial_p_interp(nsample);
  Eigen::ArrayXd partial_tt_interp(nsample);
  Eigen::ArrayXd partial_pp_interp(nsample);
  Eigen::ArrayXd partial_tp_interp(nsample);
  std::vector<int> idx(nsample);
#pragma omp parallel for schedule(dynamic, 1)
  for(int i = 0; i < nsample; ++i){
    pointing ptg(p_theta[i], p_phi[i]);
    fix_arr<int, 4> neighbours_idx;
    fix_arr<double, 4> weight;
    hb.get_interpol(ptg, neighbours_idx, weight);


    Eigen::Array4d weight_arr(weight[0], weight[1], weight[2], weight[3]);
    Eigen::Array4i neighbours_idx_arr(neighbours_idx[0],neighbours_idx[1],neighbours_idx[2],neighbours_idx[3]);

    smoothed_map_interp(i) = (smoothed_map(neighbours_idx_arr).array()*weight_arr).sum();
    partial_t_interp(i)    = (partial_t(neighbours_idx_arr).array()*weight_arr).sum();
    partial_p_interp(i)    = (partial_p(neighbours_idx_arr).array()*weight_arr).sum();
    partial_tt_interp(i)   = (partial_tt(neighbours_idx_arr).array()*weight_arr).sum();
    partial_tp_interp(i)   = (partial_tp(neighbours_idx_arr).array()*weight_arr).sum();
    partial_pp_interp(i)   = (partial_pp(neighbours_idx_arr).array()*weight_arr).sum();


    idx[i] = hb.ang2pix(ptg);
  }




  // Monopole
  // gain
  map_tod += g_vector * smoothed_map_interp.matrix().transpose();

  // beam width
  Eigen::Vector2d delta_s = s_vector.array() - s_bar;
  map_tod += delta_s * s_bar * (partial_tt_interp + partial_pp_interp).matrix().transpose();

  // Dipole
  map_tod += x_vector * ( cospsi*partial_p_interp - sinpsi*partial_t_interp).matrix().transpose();
  map_tod += y_vector * (-cospsi*partial_t_interp - sinpsi*partial_p_interp).matrix().transpose();

  // Quadrupole
  Eigen::ArrayXd partial_xx = cospsi*cospsi*partial_pp_interp +
                              sinpsi*sinpsi*partial_tt_interp -
                              sin2psi*partial_tp_interp;
  Eigen::ArrayXd partial_yy = sinpsi*sinpsi*partial_pp_interp +
                              cospsi*cospsi*partial_tt_interp +
                              sin2psi*partial_tp_interp;
  Eigen::ArrayXd partial_xy = 0.5*(
                                sin2psi*(partial_tt_interp-partial_pp_interp) -
                                2*cos2psi*partial_tp_interp
                              );

  Eigen::ArrayXd p_xx_minus_p_yy = partial_xx - partial_yy;

  map_tod += 0.5*p_vector * s_bar*s_bar * p_xx_minus_p_yy.matrix().transpose();
  map_tod +=     c_vector * s_bar*s_bar * partial_xy.matrix().transpose();

  return tod;
}

void test_speed(){
  Eigen::initParallel();
  std::cout << Eigen::nbThreads() << std::endl;
  struct timespec time1, time2;
  int size = 1000;
  Eigen::MatrixXd m1 = Eigen::MatrixXd::Random(size,size);
  Eigen::MatrixXd m2 = Eigen::MatrixXd::Random(size,size);
  Eigen::MatrixXd m3(size, size);
  std::cout << "Start!" << std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1; ++i){
    // std::cout << i << std::endl;
    m3 = m1*m2;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  auto duration_s  = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();
  std::cout << duration_ns << "us" << std::endl;
  std::cout << duration_us << "ns" << std::endl;
  std::cout << duration_ms << "ms" << std::endl;
  std::cout << duration_s << "s" << std::endl;
}
