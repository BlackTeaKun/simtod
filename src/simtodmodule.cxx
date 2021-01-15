#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define PY_ARRAY_UNIQUE_SYMBOL MyAPI

#include <Python.h>
#include "numpy/arrayobject.h"
#include "modulefuncs.hpp"
#include <iostream>

PyMethodDef MyMethods[] = {
    {"simtod", simtod, METH_VARARGS, "TOD simulation. (beam, maps, theta, phi, psi)"},
    {nullptr, nullptr, 0, nullptr}};

PyModuleDef simtodmodule = {
    PyModuleDef_HEAD_INIT,
    "simtod",
    nullptr,
    -1,
    MyMethods};

// The Whole Module
PyMODINIT_FUNC PyInit__simtod(void)
{
    import_array();
    PyObject *mymodule = PyModule_Create(&simtodmodule);
    if(mymodule == nullptr) return nullptr;
    if(PyType_Ready(&MapMakingType) < 0) return nullptr;

    Py_INCREF(&MapMakingType);
    if (PyModule_AddObject(mymodule, "MapMaking", reinterpret_cast<PyObject*>(&MapMakingType)) < 0) {
        Py_DECREF(&MapMakingType);
        Py_DECREF(mymodule);
        return nullptr;
    }
    return mymodule;
}
