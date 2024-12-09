//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
%module pyarmnn_version

%include "std_string.i"

%{
#define SWIG_FILE_WITH_INIT
#include "armnn/Version.hpp"
%}

%{
    std::string GetVersion()
    {
        return ARMNN_VERSION;
    };

    std::string GetMajorVersion()
    {
        return STRINGIFY_VALUE(ARMNN_MAJOR_VERSION);
    };

    std::string GetMinorVersion()
    {
        return STRINGIFY_VALUE(ARMNN_MINOR_VERSION);
    };
%}
%feature("docstring",
"
    Returns Arm NN library full version: MAJOR + MINOR + INCREMENTAL.

    Returns:
        str: Full version of Arm NN installed.

") GetVersion;
std::string GetVersion();

%feature("docstring",
"
    Returns Arm NN library major version.

    Returns:
        str: Major version of Arm NN installed.

") GetMajorVersion;
std::string GetMajorVersion();

%feature("docstring",
"
    Returns Arm NN library minor version.

    Returns:
        str: Minor version of Arm NN installed.

") GetMinorVersion;
std::string GetMinorVersion();
