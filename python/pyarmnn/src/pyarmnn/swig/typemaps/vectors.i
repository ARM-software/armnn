//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%inline %{
//-------------------------from_python_to_cpp-----------------------------
    int from_python_to_cpp(PyObject *obj, long* val) {
        return SWIG_AsVal_long(obj, val);
    }

    int from_python_to_cpp(PyObject *obj, int* val) {
        return SWIG_AsVal_int(obj, val);
    }

    int from_python_to_cpp(PyObject *obj, unsigned int* val) {
        return SWIG_AsVal_unsigned_SS_int(obj, val);
    }

    int from_python_to_cpp(PyObject *obj, unsigned short* val) {
        return SWIG_AsVal_unsigned_SS_short(obj, val);
    }

    int from_python_to_cpp(PyObject *obj, float* val) {
        return SWIG_AsVal_float(obj, val);
    }

    int from_python_to_cpp(PyObject *obj, double* val) {
        return SWIG_AsVal_double(obj, val);
    }
#ifdef SWIG_LONG_LONG_AVAILABLE
    int from_python_to_cpp(PyObject *obj, unsigned long long* val) {
        return SWIG_AsVal_unsigned_SS_long_SS_long(obj, val);
    }

    int from_python_to_cpp(PyObject *obj, long long* val) {
        return SWIG_AsVal_long_SS_long(obj, val);
    }
#endif

    int from_python_to_cpp(PyObject *obj, unsigned long* val) {
        return SWIG_AsVal_unsigned_SS_long(obj, val);
    }

    int from_python_to_cpp(PyObject *obj, short* val) {
        return SWIG_AsVal_short(obj, val);
    }
//-------------------------from_cpp_to_python-----------------------------
    PyObject* from_cpp_to_python(long& val){
        return PyLong_FromLong(val);
    }

    PyObject* from_cpp_to_python(unsigned long& val){
        return PyLong_FromUnsignedLong(val);
    }
#ifdef SWIG_LONG_LONG_AVAILABLE
    PyObject* from_cpp_to_python(long long& val){
        return PyLong_FromLongLong(val);
    }

    PyObject* from_cpp_to_python(unsigned long long& val){
        return PyLong_FromUnsignedLongLong(val);
    }
#endif

    PyObject* from_cpp_to_python(int& val){
        return PyLong_FromLong(static_cast<long>(val));
    }

    PyObject* from_cpp_to_python(unsigned int& val){
        return PyLong_FromUnsignedLong(static_cast<unsigned long>(val));
    }

    PyObject* from_cpp_to_python(unsigned short& val){
        return PyLong_FromUnsignedLong(static_cast<unsigned long>(val));
    }

    PyObject* from_cpp_to_python(float& val){
        return PyFloat_FromDouble(static_cast<double>(val));
    }

    PyObject* from_cpp_to_python(double& val){
        return PyFloat_FromDouble(val);
    }

    template<class U, class V>
    PyObject* from_cpp_to_python(std::pair<U, V>& pair){

        PyObject* first = from_cpp_to_python(pair.first);
        PyObject* second = from_cpp_to_python(pair.second);

        PyObject* localTuple = PyTuple_New(2);

        if (!localTuple) {
            Py_XDECREF(localTuple);
            return PyErr_NoMemory();
        }

        PyTuple_SetItem(localTuple, 0, first);
        PyTuple_SetItem(localTuple, 1, second);

        return localTuple;
    }

    template<class K, class V>
    static int from_python_to_cpp(PyObject* tuple, std::pair<K,V>* out) {

        if (PyTuple_Check(tuple)) {

            auto size = PyTuple_Size(tuple);

            if (size != 2) {
                return SWIG_ValueError;
            }

            PyObject* firstPy = PyTuple_GetItem(tuple, 0);
            PyObject* secondPy = PyTuple_GetItem(tuple, 1);

            if (!SWIG_IsOK(from_python_to_cpp(firstPy, &out->first))) {
                return SWIG_TypeError;
            }

            if (!SWIG_IsOK(from_python_to_cpp(secondPy, &out->second))) {
                return SWIG_TypeError;
            }

        } else {
            return SWIG_TypeError;
        }

        return SWIG_OK;
    }
//---------------std::vector <-> python list ---------------------
    template<class T>
    static PyObject* from_vector_to_python(std::vector<T>* input) {
        Py_ssize_t size = input->size();
        PyObject* localList = PyList_New(size);

        if (!localList) {
            Py_XDECREF(localList);
            return PyErr_NoMemory();
        }

        for(Py_ssize_t i = 0; i < size; ++i) {

            PyObject* obj = from_cpp_to_python(input->at(i));

            PyList_SET_ITEM(localList, i, obj);
        }
        return localList;
    }

    template<class T>
    int from_python_to_vector(PyObject* seq, std::vector<T>& out) {
        Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);

        for(Py_ssize_t i=0; i < size; i++) {
            PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
            if(!item) {
                PyErr_SetString(PyExc_TypeError, "Failed to read data from given sequence");

                return SWIG_NullReferenceError;
            }

            T element;
            int res = from_python_to_cpp(item, &element);
            if (!SWIG_IsOK(res)) {
                PyObject* itemRepr = PyObject_Repr(item);
                PyObject* itemStrObj = PyUnicode_AsEncodedString(itemRepr, "utf-8", "replace");
                const char* itemStr = PyBytes_AS_STRING(itemStrObj);

                auto pythonType = Py_TYPE(item)->tp_name;

                PyErr_Format(PyExc_TypeError, "Failed to convert python input value %s of type '%s' to C type '%s'", itemStr, pythonType, typeid(T).name());
                Py_XDECREF(itemStrObj);
                Py_XDECREF(itemRepr);
                Py_DECREF(seq);
                return SWIG_TypeError;
            }
            out.push_back(element);
        }
        return SWIG_OK;
    }

%}

%define %list_to_vector(TYPEMAP...)

// this typemap works for struct argument set
    %typemap(in) TYPEMAP* (TYPEMAP tmp) {
        if (PySequence_Check($input)) {

            if (from_python_to_vector($input, tmp) < 0) {
                SWIG_fail;
            }

            $1 = &tmp;

        } else {
            PyErr_SetString(PyExc_TypeError, "Argument value object does not provide sequence protocol, implement __getitem__() method.");
            SWIG_fail;
        }
    }

// this typemap works for constructor
    %typemap(in) TYPEMAP {
        if (PySequence_Check($input)) {
            if (from_python_to_vector($input, $1) < 0){
                SWIG_fail;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument value object does not provide sequence protocol, implement __getitem__() method.");
            SWIG_fail;
        }
    }

// this typemap works for struct argument get

    %typemap(out) TYPEMAP* {
        $result = from_vector_to_python($1);
    }

// this typemap works for overloaded methods and ctors
    %typemap(typecheck) (TYPEMAP) {
        $1 = PySequence_Check($input) ? 1 : 0;
    }

%enddef

%define %list_to_vector_clear(TYPEMAP...)
    %typemap(in) (TYPEMAP);
    %typemap(in) TYPEMAP* (TYPEMAP tmp);
    %typemap(typecheck) (TYPEMAP);
    %typemap(out) TYPEMAP*;
%enddef

