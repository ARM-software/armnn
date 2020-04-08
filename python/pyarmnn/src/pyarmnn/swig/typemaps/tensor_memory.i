//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%define %mutable_memory(TYPEMAP)
    %typemap(in) (TYPEMAP) {
      int res; void *buf = 0;
      Py_buffer view;
      res = PyObject_GetBuffer($input, &view, PyBUF_WRITABLE);
      buf = view.buf;
      PyBuffer_Release(&view);
      if (res < 0) {
        PyErr_Clear();
        %argument_fail(res, "(TYPEMAP)", $symname, $argnum);
      }
      $1 = buf;
    }

    %typemap(typecheck) (TYPEMAP) {
        $1 = PyObject_CheckBuffer($input) || PyTuple_Check($input) ? 1 : 0;
    }
%enddef

%define %clear_mutable_memory(TYPEMAP)
    %typemap(in) (TYPEMAP);
    %typemap(typecheck) (TYPEMAP);
%enddef

%define %const_memory(TYPEMAP)
    %typemap(in) (TYPEMAP) {
      int res; void *buf = 0;
      Py_buffer view;
      res = PyObject_GetBuffer($input, &view, PyBUF_CONTIG_RO);
      buf = view.buf;
      PyBuffer_Release(&view);
      if (res < 0) {
        PyErr_Clear();
        %argument_fail(res, "(TYPEMAP)", $symname, $argnum);
      }
      $1 = buf;
    }

    %typemap(typecheck) (TYPEMAP) {
        $1 = PyObject_CheckBuffer($input) || PyTuple_Check($input) ? 1 : 0;
    }
%enddef

%define %clear_const_memory(TYPEMAP)
    %typemap(in) (TYPEMAP);
    %typemap(typecheck) (TYPEMAP);
%enddef

