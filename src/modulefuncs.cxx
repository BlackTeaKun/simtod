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
#include "helper.hpp"

void test_speed();

PyObject *simtod(PyObject *self, PyObject *args)
{
  PyObject *beam_parameters;
  PyObject *map_arr;
  PyObject *theta, *phi, *psi;
  bool success = PyArg_ParseTuple(args, "OOOOO", &beam_parameters, &map_arr, &theta, &phi, &psi);
  if (!success){
    return nullptr;
  }
  bool is_all_npyarray = is_all_arrayobj(map_arr, theta, phi, psi);
  if (!is_all_npyarray){
    PyErr_Format(PyExc_TypeError, "The input map, theta, phi, psi must be numpy.ndarray object.");
    return nullptr;
  }

  int ndim_check = PyArray_NDIM((PyArrayObject*)map_arr)*1000 +
                   PyArray_NDIM((PyArrayObject*)theta)  *100  +
                   PyArray_NDIM((PyArrayObject*)phi)    *10   +
                   PyArray_NDIM((PyArrayObject*)psi)    *1;
  if (ndim_check != 1111){
    PyErr_Format(PyExc_ValueError, "object too deep for desired array.");
    return nullptr;
  }
  std::vector<int> size_check(3);
  size_check[0] = PyArray_SIZE((PyArrayObject*)theta);
  size_check[1] = PyArray_SIZE((PyArrayObject*)phi);
  size_check[2] = PyArray_SIZE((PyArrayObject*)psi);
  bool is_same_size = size_check[0] == size_check[1] &&
                      size_check[1] == size_check[2];
  if (!is_same_size){
    PyErr_Format(PyExc_ValueError, "theta, phi, psi should have the same size.");
    return nullptr;
  }

  // For Beam
  double dg,dx,dy,s,ds,dp,dc;
  // success = PyArg_ParseTuple(beam_parameters, "ddddddd", &dg, &dx, &dy, &s, &ds, &dp, &dc);
  success = parse_beam_para(beam_parameters, dg, dx, dy, s, ds, dp, dc);
  if(!success){
    PyErr_Format(PyExc_TypeError, "missing beam parameter");
    return nullptr;
  }
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
  npy_intp npix = get_map_npix((PyArrayObject*)map_arr);
  bool is_valid_npix = check_npix(npix);
  if(!is_valid_npix){
    PyErr_Format(PyExc_ValueError, "Invalid map data. Bad number of pixels.");
    return nullptr;
  }
  PyObject *new_map_arr = PyArray_FROM_OTF(map_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *new_theta   = PyArray_FROM_OTF(theta  , NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *new_phi     = PyArray_FROM_OTF(phi    , NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *new_psi     = PyArray_FROM_OTF(psi    , NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  double *p_map = (double *)PyArray_GETPTR1((PyArrayObject*)new_map_arr, 0);

  // Scan
  npy_intp nsample = PyArray_Size(theta);
  double *p_theta  = (double *)PyArray_GETPTR1((PyArrayObject*)new_theta , 0);
  double *p_phi    = (double *)PyArray_GETPTR1((PyArrayObject*)new_phi   , 0);
  double *p_psi    = (double *)PyArray_GETPTR1((PyArrayObject*)new_psi   , 0);

  // Do convolve;
  double *read_out = convolve(b1, b2, npix, p_map, nsample, p_theta, p_phi, p_psi);
  //test_speed();


  PyObject *tod_array = nullptr;
  npy_intp ndims[2] = {2, nsample};
  tod_array = PyArray_SimpleNewFromData(2, ndims, NPY_FLOAT64, (void *)read_out);
  PyArrayObject* temp = (PyArrayObject*) tod_array;
  PyArray_ENABLEFLAGS(temp, NPY_ARRAY_OWNDATA);

  Py_DECREF(new_map_arr);
  Py_DECREF(new_theta);
  Py_DECREF(new_phi);
  Py_DECREF(new_psi);
  return tod_array;
}

using funcptr = double *(const Beam &, const Beam &, const DerivTMaps&, size_t, double *, double *, double *);

PyObject *__deprojtod_helper(PyObject *self, PyObject *args, funcptr func)
{
  PyObject *beam_parameters;
  PyObject *map_arr;
  PyObject *theta, *phi, *psi;
  bool success = PyArg_ParseTuple(args, "OOOOO", &beam_parameters, &map_arr, &theta, &phi, &psi);
  if (!success){
    return nullptr;
  }
  bool is_all_npyarray = ArrayCheck(map_arr) && ArrayCheck(theta) && ArrayCheck(phi) && ArrayCheck(psi);
  if (!is_all_npyarray){
    PyErr_Format(PyExc_TypeError, "The input map, theta, phi, psi must be numpy.ndarray object.");
    return nullptr;
  }
  int ndim_check = PyArray_NDIM((PyArrayObject*)map_arr)*1000 +
                   PyArray_NDIM((PyArrayObject*)theta)  *100  +
                   PyArray_NDIM((PyArrayObject*)phi)    *10   +
                   PyArray_NDIM((PyArrayObject*)psi)    *1;
  if (ndim_check != 2111){
    PyErr_Format(PyExc_ValueError, "input dimension error.");
    return nullptr;
  }
  if (PyArray_DIMS((PyArrayObject*)map_arr)[0] != 6){
    PyErr_Format(PyExc_ValueError, "maps should have shape (6, npix).");
    return nullptr;
  }

  // For Beam
  double dg,dx,dy,s,ds,dp,dc;
  success = parse_beam_para(beam_parameters, dg, dx, dy, s, ds, dp, dc);
  if(!success){
    PyErr_Format(PyExc_TypeError, "missing beam parameter");
    return nullptr;
  }
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

  PyObject *new_map_arr = PyArray_FROM_OTF(map_arr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *new_theta   = PyArray_FROM_OTF(theta  , NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *new_phi     = PyArray_FROM_OTF(phi    , NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *new_psi     = PyArray_FROM_OTF(psi    , NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Data Access
  // Map
  npy_intp npix = get_map_npix((PyArrayObject*)map_arr);
  double *s_map   = (double *)PyArray_GETPTR2((PyArrayObject*)new_map_arr, 0, 0);
  double *dt_map  = (double *)PyArray_GETPTR2((PyArrayObject*)new_map_arr, 1, 0);
  double *dp_map  = (double *)PyArray_GETPTR2((PyArrayObject*)new_map_arr, 2, 0);
  double *dtt_map = (double *)PyArray_GETPTR2((PyArrayObject*)new_map_arr, 3, 0);
  double *dpp_map = (double *)PyArray_GETPTR2((PyArrayObject*)new_map_arr, 4, 0);
  double *dtp_map = (double *)PyArray_GETPTR2((PyArrayObject*)new_map_arr, 5, 0);

  DerivTMaps maps = {s_map, dt_map, dp_map, dtt_map, dpp_map, dtp_map, (int)npix};

  // Scan
  npy_intp nsample = PyArray_Size(theta);
  double *p_theta  = (double *)PyArray_GETPTR1((PyArrayObject*)new_theta , 0);
  double *p_phi    = (double *)PyArray_GETPTR1((PyArrayObject*)new_phi   , 0);
  double *p_psi    = (double *)PyArray_GETPTR1((PyArrayObject*)new_psi   , 0);

  // Do convolve;
  double *read_out = func(b1, b2, maps, nsample, p_theta, p_phi, p_psi);


  PyObject *tod_array = nullptr;
  npy_intp ndims[2] = {2, nsample};
  tod_array = PyArray_SimpleNewFromData(2, ndims, NPY_FLOAT64, (void *)read_out);
  PyArrayObject* temp = (PyArrayObject*) tod_array;
  PyArray_ENABLEFLAGS(temp, NPY_ARRAY_OWNDATA);

  Py_DECREF(new_map_arr);
  Py_DECREF(new_theta);
  Py_DECREF(new_phi);
  Py_DECREF(new_psi);
  return tod_array;
}

PyObject *deprojtod(PyObject *self, PyObject *args){
    return __deprojtod_helper(self, args, template_tod);
}

PyObject *deprojtod_interp(PyObject *self, PyObject *args){
    return __deprojtod_helper(self, args, template_tod_interp);
}
