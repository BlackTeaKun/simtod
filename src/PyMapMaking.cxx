#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT
#define PY_ARRAY_UNIQUE_SYMBOL MyAPI

#define OBJ2ARR(o) reinterpret_cast<PyArrayObject*>(o)
#define TODOUBLEP(o) reinterpret_cast<double *>(o)

#include <Python.h>
#include <iostream>
#include <utility>
#include "structmember.h"

#include "numpy/arrayobject.h"
#include "Eigen/Dense"

#include "modulefuncs.hpp"
#include "MapMaking.hpp"

#include "helper.hpp"

inline bool is_complex(PyArrayObject *arr)
{
  int typenum = PyArray_TYPE(arr);
  if (typenum==NPY_CDOUBLE || typenum == NPY_CFLOAT || typenum==NPY_CLONGDOUBLE){
    return true;
  }
  return false;
}

struct PyMapMaking{
    PyObject_HEAD
    int nside;
    int npix;
    MapMaking *mapmaking;
};

std::pair<double *, double *> fetch_data(PyArrayObject *arr)
{
    int ndims = PyArray_NDIM(arr);
    if( ndims == 1 ){
        double *data = (double *)PyArray_GETPTR1(arr, 0);
        return std::make_pair(data, nullptr);
    }
    else if(ndims == 2){
        double *data1 = (double *)PyArray_GETPTR2(arr, 0, 0);
        double *data2 = (double *)PyArray_GETPTR2(arr, 1, 0);
        return std::make_pair(data1, data2);
    }
    return std::make_pair(nullptr, nullptr);
}

// Constructor (__new__ in python)
PyObject *
MapMaking_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyMapMaking *self;
    self = (PyMapMaking *) type->tp_alloc(type, 0);
    if (self != nullptr) {
        self->nside  = 0;
        self->npix   = 0;
        self->mapmaking = nullptr;
    }
    return (PyObject *) self;
}

// Destructor
void
MapMaking_dealloc(PyMapMaking *self)
{
    delete self->mapmaking;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

// Initialization (__init__ in python)
int
MapMaking_init(PyMapMaking *self, PyObject *args)
{
    bool is_complex = false;
    if (!PyArg_ParseTuple(args, "i|p", &self->nside, &is_complex))
        return -1;
    self->mapmaking = new MapMaking(self->nside, is_complex);
    self->npix = self->nside*self->nside*12;
    return 0;
}

// Register class member variable
PyMemberDef MapMaking_members[] = {
    {(char*)"nside", T_INT, offsetof(PyMapMaking, nside), 0, (char*)"HealPix nside"},
    {nullptr}  /* Sentinel */
};

// Definitions of class member function

PyObject *
MapMaking_add_Scan(PyMapMaking *self, PyObject* args)
{
    PyObject *p_tod, *p_theta, *p_phi, *p_psi;
    bool success = PyArg_ParseTuple(args, "OOOO", &p_tod, &p_theta, &p_phi, &p_psi);
    if (!success){
        return nullptr;
    }
    bool is_all_array = is_all_arrayobj(p_theta, p_phi);
    if(!is_all_array){
      PyErr_Format(PyExc_TypeError, "The input theta, phi must be numpy.ndarray object.");
      return nullptr;
    }

    double *tod1, *tod2, *theta, *phi, *psi;
    tod1 = tod2 = theta = phi = psi = nullptr;
    int nsample = PyArray_Size(p_theta);
    PyObject *new_theta = nullptr, *new_phi = nullptr, *new_psi = nullptr;
    PyObject *new_tods = nullptr;

    new_theta   = PyArray_FROM_OTF(p_theta  , NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    new_phi     = PyArray_FROM_OTF(p_phi    , NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    bool istpsame = is_all_same_size_lastaxis(new_theta, new_phi);
    if(!istpsame){
        PyErr_Format(PyExc_ValueError, "The input theta, phi should have same size.");
        Py_XDECREF(new_theta);
        Py_XDECREF(new_phi);
        return nullptr;
    }
    theta = TODOUBLEP(PyArray_GETPTR1(OBJ2ARR(new_theta), 0));
    phi   = TODOUBLEP(PyArray_GETPTR1(OBJ2ARR(new_phi), 0));
    if(p_psi != Py_None){
      new_psi     = PyArray_FROM_OTF(p_psi    , NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
      bool ispsisame = is_all_same_size_lastaxis(new_psi, new_theta, new_phi);
      if(!ispsisame){
        PyErr_Format(PyExc_ValueError, "The input theta, phi, psi should have same size.");
        Py_XDECREF(new_psi);
        return nullptr;
      }
      psi   = TODOUBLEP(PyArray_GETPTR1(OBJ2ARR(new_psi), 0));
    }

    if(p_tod != Py_None){
      bool tod_iscomplex = is_complex(OBJ2ARR(p_tod));
      NPY_TYPES tod_datatype = tod_iscomplex ? NPY_CDOUBLE : NPY_DOUBLE;

      new_tods = PyArray_FROM_OTF(p_tod, tod_datatype, NPY_ARRAY_IN_ARRAY);
      bool istodsame = is_all_same_size_lastaxis(new_tods, new_theta, new_phi);
      if(!istodsame){
        PyErr_Format(PyExc_ValueError, "The input tods, theta, phi, psi should have same size at last axis.");
        Py_XDECREF(new_tods);
        return nullptr;
      }
      auto tod_data_pair = fetch_data(OBJ2ARR(new_tods));
      tod1 = tod_data_pair.first;
      tod2 = tod_data_pair.second;
    }
    Scan_data scan(nsample, tod1, tod2, theta, phi, psi);
    self->mapmaking->add_Scan(scan);

    Py_XDECREF(new_theta);
    Py_XDECREF(new_phi);
    Py_XDECREF(new_psi);
    Py_RETURN_NONE;
}

PyObject *
MapMaking_get_Map(PyMapMaking *self, PyObject* Py_UNUSED(ignored))
{
    MapMaking::RowMajorDM outmap = self->mapmaking->get_map();
    int ndata = outmap.size();
    double *raw_data = new double[ndata];
    Eigen::Map<Eigen::RowVectorXd> wrapper(raw_data, ndata);
    wrapper.array() = Eigen::Map<Eigen::RowVectorXd>(outmap.data(), ndata).array();

    PyObject *map_array = nullptr;
    npy_intp ndims[2] = {outmap.rows(), outmap.cols()};
    if(self->mapmaking->is_complex){
        ndims[1] /= 2;
    }
    NPY_TYPES datatype = (self->mapmaking->is_complex) ? NPY_CDOUBLE : NPY_FLOAT64;
    map_array = PyArray_SimpleNewFromData(2, ndims, datatype, reinterpret_cast<void*>(raw_data));
    PyArray_ENABLEFLAGS(OBJ2ARR(map_array), NPY_ARRAY_OWNDATA);
    return map_array;
}

PyObject *
MapMaking_get_Hitmap(PyMapMaking *self, PyObject* Py_UNUSED(ignored))
{
    Eigen::RowVectorXi outmap = self->mapmaking->get_hitmap();
    int32_t *raw_data   = new int32_t[self->npix];
    Eigen::Map<Eigen::RowVectorXi> wrapper(raw_data, 1, self->npix);
    wrapper.array() = outmap.array();

    PyObject *map_array = nullptr;
    npy_intp ndims = self->npix;
    map_array = PyArray_SimpleNewFromData(1, &ndims, NPY_INT32, reinterpret_cast<void*>(raw_data));
    PyArray_ENABLEFLAGS(OBJ2ARR(map_array), NPY_ARRAY_OWNDATA);
    return map_array;
}

// Register class member functions
PyMethodDef MapMaking_methods[] = {
    {"add_Scan", (PyCFunction) MapMaking_add_Scan, METH_VARARGS, "Add scan strategy, (tod, theta, phi, psi)" },
    {"get_Map", (PyCFunction) MapMaking_get_Map, METH_NOARGS, "Get skymaps." },
    {"get_Hitmap", (PyCFunction) MapMaking_get_Hitmap, METH_NOARGS, "Get hitmap." },
    {nullptr}  /* Sentinel */
};

// Register the class
// For C++ the order is importrant
PyTypeObject MapMakingType = {
    PyVarObject_HEAD_INIT(NULL, 0)
     "_simtod.MapMaking", //.tp_name =

     sizeof(PyMapMaking), //.tp_basicsize =
     0, //.tp_itemsize =

     (destructor) MapMaking_dealloc, //.tp_dealloc =
     0, //.tp_print =
     0, //.tp_getattr =
     0, //.tp_setattr =
     0, //.tp_as_async=

     0, //.tp_repr =

     0, //.tp_as_number =
     0, //.tp_as_sequence =
     0, //.tp_as_mapping =

     0, //.tp_hash =
     0, //.tp_call =
     0, //.tp_str  =
     0, //.tp_getattro =
     0, //.tp_setattro =
     0, //.tp_as_buffer =
     Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, //.tp_flags =
     "MapMaking objects", //.tp_doc =
     0, //.tp_traverse =
     0, //.tp_clear =
     0, //.tp_richcompare =
     0, //.tp_weaklistoffset =
     0, //.tp_iter =
     0, //.tp_iternext =
     MapMaking_methods, //.tp_methods =
     MapMaking_members, //.tp_members =
     0, //.tp_getset =
     0, //.tp_base =
     0, //.tp_dict =
     0, //.tp_descr_get =
     0, //.tp_descr_set =
     0, //.tp_dictoffset =
     (initproc) MapMaking_init, //.tp_init =
     0, //.tp_alloc =
     MapMaking_new, //.tp_new =
};

