//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandLineProcessor.hpp"

#include <boost/program_options.hpp>
#include <iostream>

namespace armnn
{
namespace gatordmock
{

bool CommandLineProcessor::ProcessCommandLine(int argc, char *argv[])
{
    namespace po = boost::program_options;
    po::options_description desc("Options");
    try
    {
        desc.add_options()
                ("help,h", "Display help messages")
                ("file,f",  po::value<std::string>(&m_File),
                                 "The path to the file that contains instructions for the mock gatord")
                ("namespace,n", po::value<std::string>(&m_UdsNamespace)->default_value("gatord_namespace"),
                                "The Unix domain socket namespace this server will bind to.\n"
                                "This will always be prepended with \\0 to use the abstract namespace")
                ("echo,e", po::bool_switch(&m_Echo)->default_value(false),
                                "Echo packets sent and received to stdout. Disabled by default.\n");
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fatal internal error: [" << e.what() << "]" << std::endl;
        return false;
    }

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help"))
        {
            std::cout << "Simulate a Gatord server to interact with ArmNN external profiling." << std::endl;
            std::cout << std::endl;
            std::cout << desc << std::endl;
            return false;
        }
        // Currently the file parameter is mandatory.
        if (!vm.count("file"))
        {
            std::cout << std::endl << "*** Expected --file or -f parameter." << std::endl;
            std::cout << std::endl;
            std::cout << desc << std::endl;
            return false;
        }
        po::notify(vm);
    }
    catch (const po::error& e)
    {
        std::cerr << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
        return false;
    }

    return true;
}

} // namespace gatordmock

} // namespace armnn