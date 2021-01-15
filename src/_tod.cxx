#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT
#define PY_ARRAY_UNIQUE_SYMBOL MyAPI

#include <Python.h>
#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include "numpy/arrayobject.h"
#include "Beam.hpp"
#include "convolve.hpp"
void test_speed();

inline double arcmin2rad(double _arcmin){
  double res = _arcmin / 60 * 0.017453292519943295;
  return res;
}
inline double fwhm2sigma(double _fwhm){
  double res = _fwhm / 2.3548200450309493;
  return res;
}

PyObject *simtod(PyObject *self, PyObject *args)
{
  //import_array();
  PyObject *beam_parameters;
  PyObject *map_arr;
  PyObject *theta, *phi, *psi;
  PyArg_ParseTuple(args, "OOOOO", &beam_parameters, &map_arr, &theta, &phi, &psi);

  // For Beam
  double dg,dx,dy,s,ds,dp,dc;
  PyArg_ParseTuple(beam_parameters, "ddddddd", &dg, &dx, &dy, &s, &ds, &dp, &dc);
  double g1 = 1+dg/2, g2 = 1-dg/2;
  double x1 = dx/2, x2 = -dx/2;
  double y1 = dy/2, y2 = -dy/2;
  double s1 = s+ds/2, s2 = s-ds/2;
  double p1 = dp/2, p2 = -dp/2;
  double c1 = dc/2, c2 = -dc/2;
  // arcmin to radius
  x1 = arcmin2rad(x1);
  x2 = arcmin2rad(x2);
  y1 = arcmin2rad(y1);
  y2 = arcmin2rad(y2);
  // fwhm to sigma and arcmin to radius
  s1 = arcmin2rad(fwhm2sigma(s1));
  s2 = arcmin2rad(fwhm2sigma(s2));

  Beam b1({g1,x1,y1,s1,p1,c1}, false), b2({g2, x2, y2, s2, p2, c2}, false);

  // Data Access
  // Map
  npy_intp npix = PyArray_Size(map_arr);
  double *p_map = (double *)PyArray_GETPTR1((PyArrayObject*)map_arr, 0);

  // Scan
  npy_intp nsample = PyArray_Size(theta);
  double *p_theta = (double *)PyArray_GETPTR1((PyArrayObject*)theta, 0);
  double *p_phi= (double *)PyArray_GETPTR1((PyArrayObject*)phi, 0);
  double *p_psi = (double *)PyArray_GETPTR1((PyArrayObject*)psi, 0);

  // Do convolve;
  double *read_out = convolve(b1, b2, npix, p_map, nsample, p_theta, p_phi, p_psi);
  //test_speed();


  PyObject *tod_array = nullptr;
  npy_intp ndims[2] = {2, nsample};
  tod_array = PyArray_SimpleNewFromData(2, ndims, NPY_FLOAT64, (void *)read_out);
  PyArrayObject* temp = (PyArrayObject*) tod_array;
  PyArray_ENABLEFLAGS(temp, NPY_ARRAY_OWNDATA);
  return tod_array;
}
