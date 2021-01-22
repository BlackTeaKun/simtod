#include<Python.h>
#include"numpy/arrayobject.h"
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
