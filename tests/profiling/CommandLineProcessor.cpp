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
        desc.add_options()("help,h", "Display help messages");
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

        if (vm.count("help") || argc <= 1)
        {
            std::cout << "Simulate a Gatord server to interact with ArmNN external profiling." << std::endl;
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