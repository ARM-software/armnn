//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%include "stl.i"
%include "cstring.i"
%include "std_string.i"
%include "std_vector.i"
%include "std_unordered_set.i"
%include "std_pair.i"
%include "stdint.i"
%include "carrays.i"
%include "exception.i"
%include "typemaps.i"
%include "std_iostream.i"

%ignore *::operator=;
%ignore *::operator[];


// Define exception typemap to wrap armnn exception into python exception.

%exception{
    try {
        $action
    } catch (const armnn::Exception& e) {
        SWIG_exception(SWIG_RuntimeError, const_cast<char*>(e.what()));
    }
};

%exception __getitem__ {
    try {
        $action
    } catch (const armnn::InvalidArgumentException &e) {
        SWIG_exception(SWIG_ValueError, const_cast<char*>(e.what()));
    } catch (const std::out_of_range &e) {
        SWIG_exception(SWIG_IndexError, const_cast<char*>(e.what()));
    } catch (const std::exception &e) {
        SWIG_exception(SWIG_RuntimeError, const_cast<char*>(e.what()));
    }
};

%exception __setitem__ {
    try {
        $action
    } catch (const armnn::InvalidArgumentException &e) {
        SWIG_exception(SWIG_ValueError, const_cast<char*>(e.what()));
    } catch (const std::out_of_range &e) {
        SWIG_exception(SWIG_IndexError, const_cast<char*>(e.what()));
    } catch (const std::exception &e) {
        SWIG_exception(SWIG_RuntimeError, const_cast<char*>(e.what()));
    }
};
