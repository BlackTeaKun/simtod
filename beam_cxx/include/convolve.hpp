#include "Beam.hpp"

struct DerivTMaps{
  double *s_tmap, *dt_s_tmap, *dp_s_tmap, *dtt_s_tmap, *dpp_s_tmap, *dtp_s_tmap;
  int npix;
};

double *convolve(const Beam &b1, const Beam &b2, size_t npix, double *p_map, size_t nsample,
      double *p_theta, double *p_phi, double *p_psi);


double *template_tod(const Beam &b1, const Beam &b2, const DerivTMaps &maps, size_t nsample,
    double *p_theta, double *p_phi, double *p_psi);
