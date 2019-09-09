//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <string>

namespace armnn
{

namespace gatordmock
{

// Parses the command line to extract:
//
//

/**
 * Use Boost program options to process the command line.
 * -h or --help to print the options.
 * -n or --namespace to specify the UDS namespace that the server will be listening on.
 */
class CommandLineProcessor
{
public:
    bool ProcessCommandLine(int argc, char *argv[]);

    std::string GetUdsNamespace() { return m_UdsNamespace; }

private:
    std::string m_UdsNamespace;
};

} // namespace gatordmock

} // namespace armnn
