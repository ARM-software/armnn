//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../InferenceTestImage.hpp"

#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

namespace
{

// parses the command line to extract
// * the input image file -i the input image file path (must exist)
// * the layout -l the data layout output generated with (optional - default value is NHWC)
// * the output file -o the output raw tensor file path (must not already exist)
class CommandLineProcessor
{
public:
    bool ValidateInputFile(const std::string& inputFileName)
    {
        if (inputFileName.empty())
        {
            std::cerr << "No input file name specified" << std::endl;
            return false;
        }

        if (!boost::filesystem::exists(inputFileName))
        {
            std::cerr << "Input file [" << inputFileName << "] does not exist" << std::endl;
            return false;
        }

        if (boost::filesystem::is_directory(inputFileName))
        {
            std::cerr << "Input file [" << inputFileName << "] is a directory" << std::endl;
            return false;
        }

        return true;
    }

    bool ValidateLayout(const std::string& layout)
    {
        if (layout.empty())
        {
            std::cerr << "No layout specified" << std::endl;
            return false;
        }

        std::vector<std::string> supportedLayouts = {
            "NHWC",
            "NCHW"
        };

        auto iterator = std::find(supportedLayouts.begin(), supportedLayouts.end(), layout);
        if (iterator == supportedLayouts.end())
        {
            std::cerr << "Layout [" << layout << "] is not supported" << std::endl;
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
            std::cerr << "Output directory [" << outputPath.parent_path().c_str() << "] does not exist" << std::endl;
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
                ("infile,i", po::value<std::string>(&m_InputFileName)->required(),
                             "Input image file to generate tensor from")
                ("layout,l", po::value<std::string>(&m_Layout)->default_value("NHWC"),
                             "Output data layout, \"NHWC\" or \"NCHW\", default value NHWC")
                ("outfile,o", po::value<std::string>(&m_OutputFileName)->required(),
                              "Output raw tensor file path");
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

        if (!ValidateInputFile(m_InputFileName))
        {
            return false;
        }

        if (!ValidateLayout(m_Layout))
        {
            return false;
        }

        if (!ValidateOutputFile(m_OutputFileName))
        {
            return false;
        }

        return true;
    }

    std::string GetInputFileName() {return m_InputFileName;}
    std::string GetLayout() {return m_Layout;}
    std::string GetOutputFileName() {return m_OutputFileName;}

private:
    std::string m_InputFileName;
    std::string m_Layout;
    std::string m_OutputFileName;
};

} // namespace anonymous

int main(int argc, char* argv[])
{
    CommandLineProcessor cmdline;
    if (!cmdline.ProcessCommandLine(argc, argv))
    {
        return -1;
    }

    const std::string imagePath(cmdline.GetInputFileName());
    const std::string outputPath(cmdline.GetOutputFileName());

    // generate image tensor
    std::vector<float> imageData;
    try
    {
        InferenceTestImage testImage(imagePath.c_str());
        imageData = cmdline.GetLayout() == "NHWC"
             ? GetImageDataAsNormalizedFloats(ImageChannelLayout::Rgb, testImage)
             : GetImageDataInArmNnLayoutAsNormalizedFloats(ImageChannelLayout::Rgb, testImage);
    }
    catch (const InferenceTestImageException& e)
    {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to load image file " << imagePath << " with error: " << e.what();
        return -1;
    }

    std::ofstream imageTensorFile;
    imageTensorFile.open(outputPath, std::ofstream::out);
    if (imageTensorFile.is_open())
    {
        std::copy(imageData.begin(), imageData.end(), std::ostream_iterator<float>(imageTensorFile, " "));
        if (!imageTensorFile)
        {
            BOOST_LOG_TRIVIAL(fatal) << "Failed to write to output file" << outputPath;
            imageTensorFile.close();
            return -1;
        }
        imageTensorFile.close();
    }
    else
    {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open output file" << outputPath;
        return -1;
    }

    return 0;
}