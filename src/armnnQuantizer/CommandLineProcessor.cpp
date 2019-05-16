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

bool ValidateProvidedFile(const std::string& inputFileName)
{
    if (!boost::filesystem::exists(inputFileName))
    {
        std::cerr << "Provided file [" << inputFileName << "] does not exist" << std::endl;
        return false;
    }

    if (boost::filesystem::is_directory(inputFileName))
    {
        std::cerr << "Given file [" << inputFileName << "] is a directory" << std::endl;
        return false;
    }

    return true;
}

bool ValidateQuantizationScheme(const std::string& scheme)
{
    if (scheme.empty())
    {
        std::cerr << "No Quantization Scheme specified" << std::endl;
        return false;
    }

    std::vector<std::string> supportedSchemes = {
        "QAsymm8",
        "QSymm16"
    };

    auto iterator = std::find(supportedSchemes.begin(), supportedSchemes.end(), scheme);
    if (iterator == supportedSchemes.end())
    {
        std::cerr << "Quantization Scheme [" << scheme << "] is not supported" << std::endl;
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
                ("scheme,s", po::value<std::string>(&m_QuantizationScheme)->default_value("QAsymm8"),
                              "Quantization scheme, \"QAsymm8\" or \"QSymm16\", default value QAsymm8")
                ("csvfile,c", po::value<std::string>(&m_CsvFileName)->default_value(""),
                             "CSV file containing paths for RAW input tensors")
                ("preserve-data-type,p", po::bool_switch(&m_PreserveDataType)->default_value(false),
                              "Preserve the input and output data types")
                ("outdir,d", po::value<std::string>(&m_OutputDirectory)->required(),
                             "Directory that output file will be written to")
                ("outfile,o", po::value<std::string>(&m_OutputFileName)->required(), "ArmNN output file name");
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
            std::cout << "Convert a Fp32 ArmNN model to a quantized ArmNN model."  << std::endl;
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

    if (!armnnQuantizer::ValidateProvidedFile(m_InputFileName))
    {
        return false;
    }

    if (!ValidateQuantizationScheme(m_QuantizationScheme))
    {
        return false;
    }

    if (m_CsvFileName != "")
    {
        if (!armnnQuantizer::ValidateProvidedFile(m_CsvFileName))
        {
            return false;
        }
        else
        {
            boost::filesystem::path csvFilePath(m_CsvFileName);
            m_CsvFileDirectory = csvFilePath.parent_path().c_str();
        }

        // If CSV file is defined, create a QuantizationDataSet for specified CSV file.
        m_QuantizationDataSet = QuantizationDataSet(m_CsvFileName);
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