//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%define %optimize_typemap_out
    %typemap(out) (std::pair<armnn::IOptimizedNetwork*, std::vector<std::string>>) {
        PyObject * network = SWIG_NewPointerObj(SWIG_as_voidptr($1.first), SWIGTYPE_p_armnn__IOptimizedNetwork, SWIG_POINTER_OWN);
        $result = PyTuple_New(2);

        // Convert vector to fixed-size tuple
        std::vector<std::string> strings = $1.second;
        Py_ssize_t size = strings.size();

        // New reference. Need to Py_DECREF
        PyObject* errMsgTuple = PyTuple_New(size);

        if (!errMsgTuple) {
            Py_XDECREF(errMsgTuple);
            return PyErr_NoMemory();
        }

        for (Py_ssize_t i = 0; i < size; i++) {
            // New reference. Need to Py_DECREF
            PyObject *string = PyString_FromString(strings[i].c_str());

            if (!string) {
                Py_XDECREF(string);
                return PyErr_NoMemory();
            }
            PyTuple_SetItem(errMsgTuple, i, string);
        }

        // Create result tuple
        PyTuple_SetItem($result, 0, network);
        PyTuple_SetItem($result, 1, errMsgTuple);
    }
%enddef

%define %clear_optimize_typemap_out
    %typemap(out) (std::pair<armnn::IOptimizedNetwork*, std::vector<std::string>>)
%enddef
