//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandLineProcessor.hpp"

#define BOOST_FILESYSTEM_NO_DEPRECATED

#include <boost/program_options.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

namespace armnnQuantizer
{

bool ValidateOutputDirectory(std::string& dir)
{
    if (dir.empty())
    {
        std::cerr << "No output directory specified" << std::endl;
        return false;
    }

    if (dir[dir.length() - 1] != '/')
    {
        dir += "/";
    }

    if (!boost::filesystem::exists(dir))
    {
        std::cerr << "Output directory [" << dir << "] does not exist" << std::endl;
        return false;
    }

    if (!boost::filesystem::is_directory(dir))
    {
        std::cerr << "Given output directory [" << dir << "] is not a directory" << std::endl;
        return false;
    }

    return true;
}

bool ValidateInputFile(const std::string& inputFileName)
{
    if (!boost::filesystem::exists(inputFileName))
    {
        std::cerr << "Input file [" << inputFileName << "] does not exist" << std::endl;
        return false;
    }

    if (boost::filesystem::is_directory(inputFileName))
    {
        std::cerr << "Given input file [" << inputFileName << "] is a directory" << std::endl;
        return false;
    }

    return true;
}

bool CommandLineProcessor::ProcessCommandLine(int argc, char* argv[])
{
    namespace po = boost::program_options;

    po::options_description desc("Options");
    try
    {
        desc.add_options()
                ("help,h", "Display help messages")
                ("infile,f", po::value<std::string>(&m_InputFileName)->required(),
                             "Input file containing float 32 ArmNN Input Graph")
                ("outdir,d", po::value<std::string>(&m_OutputDirectory)->required(),
                             "Directory that output file will be written to")
                ("outfile,o", po::value<std::string>(&m_OutputFileName)->required(), "Output file name");
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

    if (!armnnQuantizer::ValidateInputFile(m_InputFileName))
    {
        return false;
    }

    if (!armnnQuantizer::ValidateOutputDirectory(m_OutputDirectory))
    {
        return false;
    }

    std::string output(m_OutputDirectory);
    output.append(m_OutputFileName);
    
    if (boost::filesystem::exists(output))
    {
        std::cerr << "Output file [" << output << "] already exists" << std::endl;
        return false;
    }

    return true;
}

} // namespace armnnQuantizer