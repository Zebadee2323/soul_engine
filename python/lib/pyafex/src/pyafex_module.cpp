#include <Python.h>
// --------------------------------------------------------------------------------------------------------------------
#include <afex/afex.hpp>

namespace
{

PyObject* placeholder_method(PyObject*, PyObject*) {
    const auto result = afex::placeholder_method();
    return PyUnicode_FromString(result.c_str());
}

PyMethodDef pyafex_methods[] = {
    {"placeholder_method", placeholder_method, METH_NOARGS, "Execute the afex placeholder method."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef pyafex_module = {
    PyModuleDef_HEAD_INIT,
    "_pyafex",
    "Native bindings for the afex library.",
    -1,
    pyafex_methods,
};

}

PyMODINIT_FUNC PyInit__pyafex() {
    return PyModule_Create(&pyafex_module);
}
