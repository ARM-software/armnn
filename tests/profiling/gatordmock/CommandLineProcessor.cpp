//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandLineProcessor.hpp"

#include <cxxopts/cxxopts.hpp>
#include <iostream>

namespace armnn
{
namespace gatordmock
{

bool CommandLineProcessor::ProcessCommandLine(int argc, char *argv[])
{
    cxxopts::Options options("GatordMock",
                             "Simulate a Gatord server to interact with ArmNN external profiling.");

    try
    {
        options.add_options()
            ("h,help", "Display help messages")
            ("f,file",
                "The path to the file that contains instructions for the mock gatord.",
                cxxopts::value<std::string>(m_File))
            ("n,namespace",
                "The Unix domain socket namespace this server will bind to.\n"
                "This will always be prepended with \\0 to use the abstract namespace",
                cxxopts::value<std::string>(m_UdsNamespace)->default_value("gatord_namespace"))
            ("e,echo",
                "Echo packets sent and received to stdout. Disabled by default. "
                "Default value = false.",
                cxxopts::value<bool>(m_Echo)->default_value("false"));
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fatal internal error: [" << e.what() << "]" << std::endl;
        return false;
    }

    try
    {
        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            return false;
        }

        // Currently the file parameter is mandatory.
        if (!result.count("file"))
        {
            std::cout << "-f/--file parameter is mandatory." << std::endl;
            return false;
        }

        // Sets bool value correctly.
        if (result.count("echo"))
        {
            m_Echo = true;
        }
    }
    catch (const cxxopts::OptionException& e)
    {
        std::cerr << e.what() << std::endl;
        return false;
    }

    return true;
}

} // namespace gatordmock

} // namespace armnn