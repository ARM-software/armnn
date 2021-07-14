//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnnUtils/Filesystem.hpp>
#include <cxxopts/cxxopts.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

namespace
{

// parses the command line to extract
// * the directory -i to look through .raw files from (must exist)
// * the name of the file -o the output CSV file path (must not already exist)
class CommandLineProcessor
{
public:
    bool ParseOptions(cxxopts::ParseResult& result)
    {
        // indir is mandatory, dir could possibly be changed.
        if (result.count("indir"))
        {
            std::string dir = result["indir"].as<std::string>();

            if (!ValidateDirectory(dir))
            {
                return false;
            }

            m_InputDirectory = dir;
        }
        else
        {
            std::cerr << "-i/--indir parameter is mandatory." << std::endl;
            return false;
        }

        // outfile is mandatory
        if (result.count("outfile"))
        {
            if (!ValidateOutputFile(result["outfile"].as<std::string>()))
            {
                return false;
            }
        }
        else
        {
            std::cerr << "-o/--outfile parameter is mandatory." << std::endl;
            return false;
        }

        if (result.count("layer-binding-id"))
        {
            if(!ValidateBindingId(result["layer-binding-id"].as<std::string>()))
            {
                return false;
            }
        }
        return true;
    }

    bool ValidateDirectory(std::string& dir)
    {
        if (dir.empty())
        {
            std::cerr << "No directory specified" << std::endl;
            return false;
        }

        if (dir[dir.length() - 1] != '/')
        {
            dir += "/";
        }

        if (!fs::exists(dir))
        {
            std::cerr << "Directory [" << dir << "] does not exist" << std::endl;
            return false;
        }

        if (!fs::is_directory(dir))
        {
            std::cerr << "Given directory [" << dir << "] is not a directory" << std::endl;
            return false;
        }

        return true;
    }

    bool ValidateOutputFile(const std::string& outputFileName)
    {
        if (outputFileName.empty())
        {
            std::cerr << "No output file name specified" << std::endl;
            return false;
        }

        if (fs::exists(outputFileName))
        {
            std::cerr << "Output file [" << outputFileName << "] already exists" << std::endl;
            return false;
        }

        if (fs::is_directory(outputFileName))
        {
            std::cerr << "Output file [" << outputFileName << "] is a directory" << std::endl;
            return false;
        }

        fs::path outputPath(outputFileName);
        if (!fs::exists(outputPath.parent_path()))
        {
            std::cerr << "Directory [" << outputPath.parent_path().c_str() << "] does not exist" << std::endl;
            return false;
        }

        return true;
    }

    bool ValidateBindingId(const std::string& id)
    {
        if (!std::all_of(id.begin(), id.end(), ::isdigit))
        {
            std::cerr << "Invalid input binding Id" << std::endl;
            return false;
        }

        return true;
    }

    bool ProcessCommandLine(int argc, char* argv[])
    {
        try
        {
            cxxopts::Options options("ImageCSVFileGenerator",
                                     "Program for creating a CSV file that "
                                     "contains a list of .raw tensor files. "
                                     "These .raw tensor files can be generated using the ImageTensorGenerator");

            options.add_options()
                ("h,help", "Display help messages")
                ("i,indir",
                    "Directory that .raw files are stored in",
                    cxxopts::value<std::string>(m_InputDirectory))
                ("o,outfile",
                    "Output CSV file path",
                    cxxopts::value<std::string>(m_OutputFileName))
                ("l, layer-binding-id",
                    "Input layer binding Id, Defaults to 0",
                    cxxopts::value<std::string>(m_InputBindingId)->default_value("0"));

            auto result = options.parse(argc, argv);

            if (result.count("help"))
            {
                std::cout << options.help() << std::endl;
                return false;
            }

            // Check for mandatory parameters and validate inputs
            if(!ParseOptions(result)){
                return false;
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

        return true;
    }

    std::string GetInputDirectory() {return m_InputDirectory;}
    std::string GetOutputFileName() {return m_OutputFileName;}
    std::string GetInputBindingId() {return m_InputBindingId;}

private:
    std::string m_InputDirectory;
    std::string m_OutputFileName;
    std::string m_InputBindingId;
};

} // namespace anonymous

int main(int argc, char* argv[])
{
    CommandLineProcessor cmdline;
    if (!cmdline.ProcessCommandLine(argc, argv))
    {
        return -1;
    }

    const std::string fileFormat(".raw");

    const std::string rawDirectory(cmdline.GetInputDirectory());
    const std::string outputPath(cmdline.GetOutputFileName());
    const std::string bindingId(cmdline.GetInputBindingId());

    std::vector<fs::path> rawFiles;
    for (auto& entry : fs::directory_iterator(rawDirectory))
    {
        if (entry.path().extension().c_str() == fileFormat)
        {
            rawFiles.push_back(entry.path());
        }
    }

    if (!rawFiles.empty())
    {
        unsigned int pass = 0;
        std::ofstream refinementData;
        refinementData.open(outputPath, std::ofstream::out);
        if (refinementData.is_open())
        {
            for (auto const& raw : rawFiles)
            {
                refinementData << pass << ", " << bindingId << ", " << raw.c_str() << "\n";
                if (!refinementData)
                {
                    std::cerr << "Failed to write to output file: " << outputPath << std::endl;
                    continue;
                }
                ++pass;
            }
            refinementData.close();
        }
        else
        {
            std::cerr << "Failed to open output file: " << outputPath << std::endl;
            return -1;
        }
    }
    else
    {
        std::cerr << "No matching files with the \".raw\" extension found in the directory: "
                  << rawDirectory << std::endl;
        return -1;
    }

    return 0;
}