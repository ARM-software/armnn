//
// Copyright © 2021 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%inline %{

   static PyObject* from_model_options_to_python(std::vector<armnn::BackendOptions>* input) {
        Py_ssize_t size = input->size();
        PyObject* localList = PyList_New(size);

        if (!localList) {
            Py_XDECREF(localList);
            return PyErr_NoMemory();
        }

        for(Py_ssize_t i = 0; i < size; ++i) {

            PyObject* obj = SWIG_NewPointerObj(SWIG_as_voidptr(&input->at(i)), SWIGTYPE_p_armnn__BackendOptions, 0 |  0 );

            PyList_SET_ITEM(localList, i, obj);
        }
        return localList;
    }
%}

%define %model_options_typemap

// this typemap works for struct argument get

    %typemap(out) std::vector<armnn::BackendOptions>* {
        $result = from_model_options_to_python($1);
    }

// this typemap works for struct argument set
    %typemap(in) std::vector<armnn::BackendOptions>* {
        if (PySequence_Check($input)) {

            int res = swig::asptr($input, &$1);
            if (!SWIG_IsOK(res) || !$1) {
                SWIG_exception_fail(SWIG_ArgError(($1 ? res : SWIG_TypeError)),
                    "in method '" "OptimizerOptions_m_ModelOptions_set" "', argument " "2"" of type '" "std::vector< armnn::BackendOptions,std::allocator< armnn::BackendOptions > > *""'");
            }

        } else {
            PyErr_SetString(PyExc_TypeError, "Argument value object does not provide sequence protocol.");
            SWIG_fail;
        }
    }

%enddef

%define %model_options_clear
    %typemap(out) std::vector<armnn::BackendOptions>*;
    %typemap(in) std::vector<armnn::BackendOptions>*;
%enddef
