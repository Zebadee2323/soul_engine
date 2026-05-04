#include <Python.h>
// --------------------------------------------------------------------------------------------------------------------
#include <afex/afex.hpp>
// --------------------------------------------------------------------------------------------------------------------
#include <exception>
#include <string>
#include <vector>

namespace
{

PyObject* placeholder_method(PyObject*, PyObject*) {
    const auto result = afex::placeholder_method();
    return PyUnicode_FromString(result.c_str());
}

PyObject* builtin_feature_names(PyObject*, PyObject*) {
    const auto names = afex::builtin_feature_names();
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(names.size()));
    if (list == nullptr) {
        return nullptr;
    }

    for (auto i = std::size_t{0}; i < names.size(); ++i) {
        PyObject* name = PyUnicode_FromString(names[i].c_str());
        if (name == nullptr) {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), name);
    }

    return list;
}

bool set_python_error_from_exception() {
    try {
        throw;
    } catch (const std::invalid_argument& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown afex error.");
    }

    return false;
}

bool parse_extractor_configs(PyObject* object, std::vector<afex::ExtractorConfig>& configs) {
    if (object == nullptr || object == Py_None) {
        return true;
    }

    if (!PySequence_Check(object)) {
        PyErr_SetString(PyExc_TypeError, "extractors must be a sequence of strings or dictionaries.");
        return false;
    }

    const auto size = PySequence_Size(object);
    if (size < 0) {
        return false;
    }

    configs.reserve(static_cast<std::size_t>(size));
    for (auto i = Py_ssize_t{0}; i < size; ++i) {
        PyObject* item = PySequence_GetItem(object, i);
        if (item == nullptr) {
            return false;
        }

        auto config = afex::ExtractorConfig{};
        if (PyUnicode_Check(item)) {
            config.name = PyUnicode_AsUTF8(item);
        } else if (PyDict_Check(item)) {
            PyObject* name = PyDict_GetItemString(item, "name");
            if (name == nullptr || !PyUnicode_Check(name)) {
                Py_DECREF(item);
                PyErr_SetString(PyExc_TypeError, "extractor dictionaries must include a string 'name'.");
                return false;
            }

            config.name = PyUnicode_AsUTF8(name);
            PyObject* parameters = PyDict_GetItemString(item, "parameters");
            if (parameters == nullptr) {
                parameters = item;
            }
            if (!PyDict_Check(parameters)) {
                Py_DECREF(item);
                PyErr_SetString(PyExc_TypeError, "extractor 'parameters' must be a dictionary when provided.");
                return false;
            }

            PyObject* key = nullptr;
            PyObject* value = nullptr;
            auto position = Py_ssize_t{0};
            while (PyDict_Next(parameters, &position, &key, &value)) {
                if (!PyUnicode_Check(key)) {
                    Py_DECREF(item);
                    PyErr_SetString(PyExc_TypeError, "extractor parameter names must be strings.");
                    return false;
                }

                const auto parameter_name = std::string{PyUnicode_AsUTF8(key)};
                if (parameter_name == "name" || parameter_name == "parameters") {
                    continue;
                }

                const auto parameter_value = PyFloat_AsDouble(value);
                if (PyErr_Occurred() != nullptr) {
                    Py_DECREF(item);
                    return false;
                }

                config.parameters[parameter_name] = parameter_value;
            }
        } else {
            Py_DECREF(item);
            PyErr_SetString(PyExc_TypeError, "each extractor must be a string or dictionary.");
            return false;
        }

        Py_DECREF(item);
        configs.push_back(std::move(config));
    }

    return true;
}

PyObject* feature_result_to_python(const afex::FeatureResult& feature) {
    PyObject* dict = PyDict_New();
    if (dict == nullptr) {
        return nullptr;
    }

    PyDict_SetItemString(dict, "name", PyUnicode_FromString(feature.name.c_str()));
    PyDict_SetItemString(dict, "status", PyUnicode_FromString(feature.status == afex::FeatureStatus::Complete ? "complete" : "failed"));
    if (feature.value.has_value()) {
        PyDict_SetItemString(dict, "value", PyFloat_FromDouble(*feature.value));
    } else {
        Py_INCREF(Py_None);
        PyDict_SetItemString(dict, "value", Py_None);
    }

    PyObject* values = PyList_New(static_cast<Py_ssize_t>(feature.values.size()));
    if (values == nullptr) {
        Py_DECREF(dict);
        return nullptr;
    }

    for (auto i = std::size_t{0}; i < feature.values.size(); ++i) {
        PyList_SET_ITEM(values, static_cast<Py_ssize_t>(i), PyFloat_FromDouble(feature.values[i]));
    }

    PyDict_SetItemString(dict, "values", values);
    Py_DECREF(values);
    PyDict_SetItemString(dict, "unit", PyUnicode_FromString(feature.unit.c_str()));
    PyDict_SetItemString(dict, "note", PyUnicode_FromString(feature.note.c_str()));
    return dict;
}

PyObject* analysis_result_to_python(const afex::AnalysisResult& result) {
    PyObject* dict = PyDict_New();
    if (dict == nullptr) {
        return nullptr;
    }

    PyDict_SetItemString(dict, "sample_rate_hz", PyLong_FromUnsignedLong(result.sample_rate_hz));
    PyDict_SetItemString(dict, "channel_count", PyLong_FromUnsignedLong(result.channel_count));
    PyDict_SetItemString(dict, "sample_count", PyLong_FromSize_t(result.sample_count));
    PyDict_SetItemString(dict, "frame_count", PyLong_FromSize_t(result.frame_count));
    PyDict_SetItemString(dict, "duration_seconds", PyFloat_FromDouble(result.duration_seconds));

    PyObject* features = PyList_New(static_cast<Py_ssize_t>(result.features.size()));
    if (features == nullptr) {
        Py_DECREF(dict);
        return nullptr;
    }

    for (auto i = std::size_t{0}; i < result.features.size(); ++i) {
        PyObject* feature = feature_result_to_python(result.features[i]);
        if (feature == nullptr) {
            Py_DECREF(features);
            Py_DECREF(dict);
            return nullptr;
        }
        PyList_SET_ITEM(features, static_cast<Py_ssize_t>(i), feature);
    }

    PyDict_SetItemString(dict, "features", features);
    Py_DECREF(features);
    return dict;
}

PyObject* analyze_audio_file(PyObject*, PyObject* args, PyObject* kwargs) {
    const char* audio_file_path = nullptr;
    PyObject* extractors = Py_None;
    static char const* keywords[] = {"audio_file_path", "extractors", nullptr};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|O", const_cast<char**>(keywords), &audio_file_path, &extractors)) {
        return nullptr;
    }

    try {
        auto configs = std::vector<afex::ExtractorConfig>{};
        if (!parse_extractor_configs(extractors, configs)) {
            return nullptr;
        }

        return analysis_result_to_python(afex::analyze_file(audio_file_path, configs));
    } catch (...) {
        set_python_error_from_exception();
        return nullptr;
    }
}

PyObject* analyze_audio_file_with_yaml(PyObject*, PyObject* args, PyObject* kwargs) {
    const char* config_file_path = nullptr;
    const char* audio_file_path = nullptr;
    static char const* keywords[] = {"config_file_path", "audio_file_path", nullptr};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|z", const_cast<char**>(keywords), &config_file_path, &audio_file_path)) {
        return nullptr;
    }

    try {
        auto config = afex::load_analysis_config_yaml(config_file_path);
        if (audio_file_path != nullptr) {
            config.audio_file_path = audio_file_path;
        }

        return analysis_result_to_python(afex::analyze_file(config));
    } catch (...) {
        set_python_error_from_exception();
        return nullptr;
    }
}

PyMethodDef pyafex_methods[] = {
    {"placeholder_method", placeholder_method, METH_NOARGS, "Execute the afex placeholder method."},
    {"builtin_feature_names", builtin_feature_names, METH_NOARGS, "Return the names of all built-in afex extractors."},
    {"analyze_audio_file", reinterpret_cast<PyCFunction>(analyze_audio_file), METH_VARARGS | METH_KEYWORDS, "Analyze an audio file with selected extractors."},
    {"analyze_audio_file_with_yaml", reinterpret_cast<PyCFunction>(analyze_audio_file_with_yaml), METH_VARARGS | METH_KEYWORDS, "Analyze an audio file using an afex YAML extractor config."},
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
