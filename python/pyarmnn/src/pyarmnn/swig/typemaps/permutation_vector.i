//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%define %permutation_vector_typemap(TYPE1, TYPE2)
    %typemap(in) (TYPE1, TYPE2) {
        if (PyTuple_Check($input)) {
            PyObject* seq = $input;

            $2 = PySequence_Fast_GET_SIZE(seq);
            $1 = (unsigned int*)PyMem_RawMalloc($2*sizeof(unsigned int));


            if(!$1) {
                PyErr_NoMemory();
                SWIG_fail;
            }
            int size = (int)$2;
            for(int i=0; i < size; i++) {
                PyObject *longItem;
                // Borrowed reference. No need to Py_DECREF
                PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
                if(!item) {
                    PyErr_SetString(PyExc_TypeError, "Failed to read data from tuple");
                    SWIG_fail;
                }
                // New reference. Need to Py_DECREF
                longItem = PyNumber_Long(item);
                if(!longItem) {
                    Py_XDECREF(longItem);
                    PyErr_SetString(PyExc_TypeError, "All elements must be numbers");
                    SWIG_fail;
                }
                $1[i] = (unsigned int)PyLong_AsUnsignedLong(longItem);
                Py_XDECREF(longItem);
            }

        } else {
            PyErr_SetString(PyExc_TypeError, "Argument is not a tuple");
            SWIG_fail;
        }
    }

    %typemap(freearg) (TYPE1, TYPE2) {
        PyMem_RawFree($1);
    }
%enddef

%define %clear_permutation_vector_typemap(TYPE1, TYPE2)
    %typemap(in) (TYPE1, TYPE2);
    %typemap(freearg) (TYPE1, TYPE2);
%enddef
