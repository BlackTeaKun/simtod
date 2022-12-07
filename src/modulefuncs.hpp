#pragma once
#include <Python.h>
#include <MapMaking.hpp>
#include "structmember.h"
// Functions
PyObject *simtod(PyObject *self, PyObject *args);
PyObject *deprojtod(PyObject *self, PyObject *args);
PyObject *deprojtod_interp(PyObject *self, PyObject *args);
// Class
extern PyTypeObject MapMakingType;
