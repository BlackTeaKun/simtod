#define PY_SSIZE_T_CLEAN

#define NO_IMPORT
#define PY_ARRAY_UNIQUE_SYMBOL MyAPI

#define OBJ2ARR(o) reinterpret_cast<PyArrayObject*>(o)
#define TODOUBLEP(o) reinterpret_cast<double *>(o)

#include <Python.h>
#include <iostream>
#include "structmember.h"

#include "numpy/arrayobject.h"
#include "Eigen/Dense"

#include "modulefuncs.hpp"
#include "MapMaking.hpp"

struct PyMapMaking{
    PyObject_HEAD
    int nside;
    int npix;
    MapMaking *mapmaking;
};

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
    if (!PyArg_ParseTuple(args, "i", &self->nside))
        return -1;
    self->mapmaking = new MapMaking(self->nside);
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
    PyArg_ParseTuple(args, "OOOO", &p_tod, &p_theta, &p_phi, &p_psi);

    double *tod1, *tod2, *theta, *phi, *psi;
    tod1 = tod2 = theta = phi = psi = nullptr;
    int nsample = PyArray_Size(p_theta);
    theta = TODOUBLEP(PyArray_GETPTR1(OBJ2ARR(p_theta), 0));
    phi   = TODOUBLEP(PyArray_GETPTR1(OBJ2ARR(p_phi), 0));
    if(p_tod != Py_None){
      tod1  = TODOUBLEP(PyArray_GETPTR2(OBJ2ARR(p_tod), 0, 0));
      tod2  = TODOUBLEP(PyArray_GETPTR2(OBJ2ARR(p_tod), 1, 0));
    }
    if(p_psi != Py_None){
      psi   = TODOUBLEP(PyArray_GETPTR1(OBJ2ARR(p_psi), 0));
    }
    Scan_data scan(nsample, tod1, tod2, theta, phi, psi);
    self->mapmaking->add_Scan(scan);

    Py_RETURN_NONE;
}

PyObject *
MapMaking_get_Map(PyMapMaking *self, PyObject* Py_UNUSED(ignored))
{
    using matrix_type = MapMaking::_eigen_type;
    matrix_type outmap = self->mapmaking->get_map();
    double *raw_data = new double[self->npix*3];
    Eigen::Map<matrix_type> wrapper(raw_data, 3, self->npix);
    wrapper.array() = outmap.array();

    PyObject *map_array = nullptr;
    npy_intp ndims[2] = {3, self->npix};
    map_array = PyArray_SimpleNewFromData(2, ndims, NPY_FLOAT64, reinterpret_cast<void*>(raw_data));
    PyArray_ENABLEFLAGS(OBJ2ARR(map_array), NPY_ARRAY_OWNDATA);
    return map_array;
}

PyObject *
MapMaking_get_Hitmap(PyMapMaking *self, PyObject* Py_UNUSED(ignored))
{
    using hit_type = MapMaking::_eigen_int_type;
    hit_type outmap = self->mapmaking->get_hitmap();
    int32_t *raw_data   = new int32_t[self->npix];
    Eigen::Map<hit_type> wrapper(raw_data, 1, self->npix);
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
    .tp_name = "_simtod.MapMaking",

    .tp_basicsize = sizeof(PyMapMaking),
    .tp_itemsize = 0,

    .tp_dealloc = (destructor) MapMaking_dealloc,
    .tp_print =0,
    .tp_getattr = 0,
    .tp_setattr = 0,
    .tp_as_async= 0,

    .tp_repr = 0,

    .tp_as_number = 0,
    .tp_as_sequence = 0,
    .tp_as_mapping = 0,

    .tp_hash = 0,
    .tp_call = 0,
    .tp_str  = 0,
    .tp_getattro = 0,
    .tp_setattro = 0,
    .tp_as_buffer = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "MapMaking objects",
    .tp_traverse = 0,
    .tp_clear = 0,
    .tp_richcompare = 0,
    .tp_weaklistoffset = 0,
    .tp_iter = 0,
    .tp_iternext = 0,
    .tp_methods = MapMaking_methods,
    .tp_members = MapMaking_members,
    .tp_getset = 0,
    .tp_base = 0,
    .tp_dict = 0,
    .tp_descr_get = 0,
    .tp_descr_set = 0,
    .tp_dictoffset = 0,
    .tp_init = (initproc) MapMaking_init,
    .tp_alloc = 0,
    .tp_new = MapMaking_new,
};

