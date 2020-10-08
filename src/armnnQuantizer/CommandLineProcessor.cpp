//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommandLineProcessor.hpp"
#include <Filesystem.hpp>

#include <cxxopts/cxxopts.hpp>

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

    if (!fs::exists(dir))
    {
        std::cerr << "Output directory [" << dir << "] does not exist" << std::endl;
        return false;
    }

    if (!fs::is_directory(dir))
    {
        std::cerr << "Given output directory [" << dir << "] is not a directory" << std::endl;
        return false;
    }

    return true;
}

bool ValidateProvidedFile(const std::string& inputFileName)
{
    if (!fs::exists(inputFileName))
    {
        std::cerr << "Provided file [" << inputFileName << "] does not exist" << std::endl;
        return false;
    }

    if (fs::is_directory(inputFileName))
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

    std::vector<std::string> supportedSchemes =
    {
        "QAsymmS8",
        "QAsymmU8",
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
    try
    {
        cxxopts::Options options("ArmnnQuantizer","Convert a Fp32 ArmNN model to a quantized ArmNN model.");

        options.add_options()
            ("h,help", "Display help messages")
            ("f,infile",
                "Input file containing float 32 ArmNN Input Graph",
                cxxopts::value<std::string>(m_InputFileName))
            ("s,scheme",
                "Quantization scheme,"
                " \"QAsymmU8\" or \"QAsymmS8\" or \"QSymm16\","
                " default value QAsymmU8",
                cxxopts::value<std::string>(m_QuantizationScheme)->default_value("QAsymmU8"))
            ("c,csvfile",
                "CSV file containing paths for RAW input tensors",
                cxxopts::value<std::string>(m_CsvFileName)->default_value(""))
            ("p,preserve-data-type",
                "Preserve the input and output data types",
                cxxopts::value<bool>(m_PreserveDataType)->default_value("false"))
            ("d,outdir",
                "Directory that output file will be written to",
                cxxopts::value<std::string>(m_OutputDirectory))
            ("o,outfile",
                "ArmNN output file name",
                cxxopts::value<std::string>(m_OutputFileName));

        auto result = options.parse(argc, argv);

        if (result.count("help") > 0 || argc <= 1)
        {
            std::cout << options.help() << std::endl;
            return false;
        }

        // Check for mandatory single options.
        std::string mandatorySingleParameters[] = { "infile", "outdir", "outfile" };
        for (auto param : mandatorySingleParameters)
        {
            if (result.count(param) != 1)
            {
                std::cerr << "Parameter \'--" << param << "\' is required but missing." << std::endl;
                return false;
            }
        }
    }
    catch (const cxxopts::OptionException& e)
    {
        std::cerr << e.what() << std::endl << std::endl;
        return false;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fatal internal error: [" << e.what() << "]" << std::endl;
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
            fs::path csvFilePath(m_CsvFileName);
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

    if (fs::exists(output))
    {
        std::cerr << "Output file [" << output << "] already exists" << std::endl;
        return false;
    }

    return true;
}

} // namespace armnnQuantizer