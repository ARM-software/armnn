//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%module pyarmnn
%{
#define SWIG_FILE_WITH_INIT
#include "armnn/Types.hpp"
#include "ProfilingGuid.hpp"
%}

//typemap definitions and other common stuff
%include "standard_header.i"

//armnn api submodules
%include "modules/armnn_backend.i"
%include "modules/armnn_backend_opt.i"
%include "modules/armnn_types.i"
%include "modules/armnn_descriptors.i"
%include "modules/armnn_lstmparam.i"
%include "modules/armnn_network.i"
%include "modules/armnn_profiler.i"
%include "modules/armnn_runtime.i"
%include "modules/armnn_tensor.i"
%include "modules/armnn_types_utils.i"

// Clear exception typemap.
%exception;

