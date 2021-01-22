#pragma once
#include <Python.h>
#include "numpy/arrayobject.h"
#include <type_traits>

#define ArrayCheck(o) Py_TYPE(o)==&PyArray_Type

// Parse parameters
template <class T>
int __helper_parse_beam(PyObject *iter, T& cur_para)
{
  PyObject *item = PyIter_Next(iter);
  if (item == nullptr) return 0;
  cur_para = PyFloat_AsDouble(item);
  Py_DECREF(item);
  return 1;
}
template <class T, class... Args>
int __helper_parse_beam(PyObject *iter, T& cur_para, Args& ... paras)
{
  PyObject *item = PyIter_Next(iter);
  if (item == nullptr) return 0;
  cur_para = PyFloat_AsDouble(item);
  Py_DECREF(item);

  return __helper_parse_beam(iter, paras...);
}
template <class... Args>
int parse_beam_para(PyObject *beam, Args& ... paras)
{
  PyObject *iter = PyObject_GetIter(beam);
  if (iter == nullptr) return 0;
  int return_val = __helper_parse_beam(iter, paras...); 
  Py_DECREF(iter);
  return return_val;
}

// is all object is numpy.ndarray
template<class T>
bool is_all_arrayobj(T* a)
{
  bool is_pyobj = std::is_same<T, PyObject>::value;
  if (a==nullptr || !is_pyobj){
    return false;
  }
  return ArrayCheck(a);
}
template<class T, class ... Args>
bool is_all_arrayobj(T* a, Args* ... others)
{
  return is_all_arrayobj(a) && is_all_arrayobj(others...);
}

// numpy.ndarray length check
template<class T>
int __helper_is_all_same_size_lastaxis(T* cur)
{
  int ndim = PyArray_NDIM((PyArrayObject*) cur);
  if(ndim == 0) return -1; 
  npy_intp* dims = PyArray_DIMS((PyArrayObject*) cur);
  return dims[ndim-1];
}
template<class T, class ... Args>
int __helper_is_all_same_size_lastaxis(T* cur, Args* ... others)
{
  int cur_size    = __helper_is_all_same_size_lastaxis(cur);
  int others_size = __helper_is_all_same_size_lastaxis(others...);
  if(cur_size == others_size) return cur_size;
  return -1;
}
template<class... Args>
bool is_all_same_size_lastaxis(Args* ... others){
  return __helper_is_all_same_size_lastaxis(others...) != -1;
}

inline double arcmin2rad(double _arcmin){
  double res = _arcmin / 60 * 0.017453292519943295;
  return res;
}
inline double fwhm2sigma(double _fwhm){
  double res = _fwhm / 2.3548200450309493;
  return res;
}
inline npy_intp get_map_npix(PyArrayObject *arr){
  int ndim = PyArray_NDIM(arr);
  npy_intp *dims = PyArray_DIMS(arr);
  return dims[ndim-1];
};
inline bool check_npix(int npix){
  if(npix < 12 || npix%12 != 0) return false;
  int temp = npix / 12;
  return (temp & (temp - 1)) == 0;
}


