//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/program_options.hpp>

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

        if (!boost::filesystem::exists(dir))
        {
            std::cerr << "Directory [" << dir << "] does not exist" << std::endl;
            return false;
        }

        if (!boost::filesystem::is_directory(dir))
        {
            std::cerr << "Given directory [" << dir << "] is not a directory" << std::endl;
            return false;
        }

        return true;
    }

    bool ValidateOutputFile(std::string& outputFileName)
    {
        if (outputFileName.empty())
        {
            std::cerr << "No output file name specified" << std::endl;
            return false;
        }

        if (boost::filesystem::exists(outputFileName))
        {
            std::cerr << "Output file [" << outputFileName << "] already exists" << std::endl;
            return false;
        }

        if (boost::filesystem::is_directory(outputFileName))
        {
            std::cerr << "Output file [" << outputFileName << "] is a directory" << std::endl;
            return false;
        }

        boost::filesystem::path outputPath(outputFileName);
        if (!boost::filesystem::exists(outputPath.parent_path()))
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
        namespace po = boost::program_options;

        po::options_description desc("Options");
        try
        {
            desc.add_options()
                ("help,h", "Display help messages")
                ("indir,i", po::value<std::string>(&m_InputDirectory)->required(),
                            "Directory that .raw files are stored in")
                ("outfile,o", po::value<std::string>(&m_OutputFileName)->required(),
                              "Output CSV file path")
                ("layer-binding-id,l", po::value<std::string>(&m_InputBindingId)->default_value("0"),
                              "Input layer binding Id, Defaults to 0");
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

        if (!ValidateDirectory(m_InputDirectory))
        {
            return false;
        }

        if (!ValidateOutputFile(m_OutputFileName))
        {
            return false;
        }

        if(!ValidateBindingId(m_InputBindingId))
        {
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

    namespace fs = boost::filesystem;

    const std::string fileFormat(".raw");

    const std::string rawDirectory(cmdline.GetInputDirectory());
    const std::string outputPath(cmdline.GetOutputFileName());
    const std::string bindingId(cmdline.GetInputBindingId());

    std::vector<fs::path> rawFiles;
    for (auto& entry : boost::make_iterator_range(fs::directory_iterator(rawDirectory), {}))
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