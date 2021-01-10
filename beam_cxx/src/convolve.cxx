#include "Beam.hpp"
#include "healpix_base.h"
#include "Eigen/Dense"

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
