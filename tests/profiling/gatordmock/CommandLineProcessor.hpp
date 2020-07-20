//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <string>

namespace armnn
{

namespace gatordmock
{

/// Use Boost program options to process the command line.
/// -h or --help to print the options.
/// -n or --namespace to specify the UDS namespace that the server will be listening on.
/// -e or --echo print all sent and received packets to stdout.
/// -f or --file The path to the file that contains instructions for the mock gatord.
class CommandLineProcessor
{
public:
    bool ProcessCommandLine(int argc, char* argv[]);
    bool IsEchoEnabled()
    {
        return m_Echo;
    }

    std::string GetUdsNamespace()
    {
        return m_UdsNamespace;
    }
    std::string GetCommandFile()
    {
        return m_File;
    }

private:
    std::string m_UdsNamespace;
    std::string m_File;

    bool m_Echo;
};

}    // namespace gatordmock

}    // namespace armnn
