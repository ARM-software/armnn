//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ExecuteNetworkParams.hpp"
#include <armnn/IRuntime.hpp>

/*
 * Historically we use the ',' character to separate dimensions in a tensor shape. However, cxxopts will read this
 * an an array of values which is fine until we have multiple tensors specified. This lumps the values of all shapes
 * together in a single array and we cannot break it up again. We'll change the vector delimiter to a '.'. We do this
 * as close as possible to the usage of cxxopts to avoid polluting other possible uses.
 */
#define CXXOPTS_VECTOR_DELIMITER '.'
#include <cxxopts/cxxopts.hpp>

/// Holds and parses program options for the ExecuteNetwork application
struct ProgramOptions
{
    /// Initializes ProgramOptions by adding options to the underlying cxxopts::options object.
    /// (Does not parse any options)
    ProgramOptions();

    /// Runs ParseOptions() on initialization
    ProgramOptions(int ac, const char* av[]);

    /// Parses program options from the command line or another source and stores
    /// the values in member variables. It also checks the validity of the parsed parameters.
    /// Throws a cxxopts exception if parsing fails or an armnn exception if parameters are not valid.
    void ParseOptions(int ac, const char* av[]);

    /// Ensures that the parameters for ExecuteNetwork fit together
    void ValidateExecuteNetworkParams();

    /// Ensures that the runtime options are valid
    void ValidateRuntimeOptions();

    cxxopts::Options m_CxxOptions;
    cxxopts::ParseResult m_CxxResult;

    ExecuteNetworkParams m_ExNetParams;
    armnn::IRuntime::CreationOptions m_RuntimeOptions;
};