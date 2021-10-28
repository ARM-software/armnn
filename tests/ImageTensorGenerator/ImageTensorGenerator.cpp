//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ImageTensorGenerator.hpp"
#include "../InferenceTestImage.hpp"
#include <armnn/Logging.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnnUtils/Filesystem.hpp>

#include <cxxopts/cxxopts.hpp>

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
    bool ParseOptions(cxxopts::ParseResult& result)
    {
        // infile is mandatory
        if (result.count("infile"))
        {
            if (!ValidateInputFile(result["infile"].as<std::string>()))
            {
                return false;
            }
        }
        else
        {
            std::cerr << "-i/--infile parameter is mandatory." << std::endl;
            return false;
        }

        // model-format is mandatory
        if (!result.count("model-format"))
        {
            std::cerr << "-f/--model-format parameter is mandatory." << std::endl;
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

        if (result.count("layout"))
        {
            if(!ValidateLayout(result["layout"].as<std::string>()))
            {
                return false;
            }
        }

        return true;
    }

    bool ValidateInputFile(const std::string& inputFileName)
    {
        if (inputFileName.empty())
        {
            std::cerr << "No input file name specified" << std::endl;
            return false;
        }

        if (!fs::exists(inputFileName))
        {
            std::cerr << "Input file [" << inputFileName << "] does not exist" << std::endl;
            return false;
        }

        if (fs::is_directory(inputFileName))
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

        std::vector<std::string> supportedLayouts = { "NHWC", "NCHW" };

        auto iterator = std::find(supportedLayouts.begin(), supportedLayouts.end(), layout);
        if (iterator == supportedLayouts.end())
        {
            std::cerr << "Layout [" << layout << "] is not supported" << std::endl;
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
            std::cerr << "Output directory [" << outputPath.parent_path().c_str() << "] does not exist" << std::endl;
            return false;
        }

        return true;
    }

    bool ProcessCommandLine(int argc, char* argv[])
    {
        cxxopts::Options options("ImageTensorGenerator",
                                 "Program for pre-processing a .jpg image "
                                 "before generating a .raw tensor file from it.");

        try
        {
            options.add_options()
                ("h,help", "Display help messages")
                ("i,infile",
                    "Input image file to generate tensor from",
                    cxxopts::value<std::string>(m_InputFileName))
                ("f,model-format",
                    "Format of the intended model file that uses the images."
                    "Different formats have different image normalization styles."
                    "If unset, defaults to tflite."
                    "Accepted value (tflite)",
                    cxxopts::value<std::string>(m_ModelFormat)->default_value("tflite"))
                ("o,outfile",
                    "Output raw tensor file path",
                    cxxopts::value<std::string>(m_OutputFileName))
                ("z,output-type",
                    "The data type of the output tensors."
                    "If unset, defaults to \"float\" for all defined inputs. "
                    "Accepted values (float, int, qasymms8 or qasymmu8)",
                    cxxopts::value<std::string>(m_OutputType)->default_value("float"))
                ("new-width",
                    "Resize image to new width. Keep original width if unspecified",
                    cxxopts::value<std::string>(m_NewWidth)->default_value("0"))
                ("new-height",
                    "Resize image to new height. Keep original height if unspecified",
                    cxxopts::value<std::string>(m_NewHeight)->default_value("0"))
                ("l,layout",
                    "Output data layout, \"NHWC\" or \"NCHW\", default value NHWC",
                    cxxopts::value<std::string>(m_Layout)->default_value("NHWC"));
        }
        catch (const std::exception& e)
        {
            std::cerr << options.help() << std::endl;
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

        return true;
    }

    std::string GetInputFileName() {return m_InputFileName;}
    armnn::DataLayout GetLayout()
    {
        if (m_Layout == "NHWC")
        {
            return armnn::DataLayout::NHWC;
        }
        else if (m_Layout == "NCHW")
        {
            return armnn::DataLayout::NCHW;
        }
        else
        {
            throw armnn::Exception("Unsupported data layout: " + m_Layout);
        }
    }
    std::string GetOutputFileName() {return m_OutputFileName;}
    unsigned int GetNewWidth() {return static_cast<unsigned int>(std::stoi(m_NewWidth));}
    unsigned int GetNewHeight() {return static_cast<unsigned int>(std::stoi(m_NewHeight));}
    SupportedFrontend GetModelFormat()
    {
        if (m_ModelFormat == "tflite")
        {
            return SupportedFrontend::TFLite;
        }
        else
        {
            throw armnn::Exception("Unsupported model format" + m_ModelFormat);
        }
    }
    armnn::DataType GetOutputType()
    {
        if (m_OutputType == "float")
        {
            return armnn::DataType::Float32;
        }
        else if (m_OutputType == "int")
        {
            return armnn::DataType::Signed32;
        }
        else if (m_OutputType == "qasymm8" || m_OutputType == "qasymmu8")
        {
            return armnn::DataType::QAsymmU8;
        }
        else if (m_OutputType == "qasymms8")
        {
            return armnn::DataType::QAsymmS8;
        }
        else
        {
            throw armnn::Exception("Unsupported input type" + m_OutputType);
        }
    }

private:
    std::string m_InputFileName;
    std::string m_Layout;
    std::string m_OutputFileName;
    std::string m_NewWidth;
    std::string m_NewHeight;
    std::string m_ModelFormat;
    std::string m_OutputType;
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
    const SupportedFrontend& modelFormat(cmdline.GetModelFormat());
    const armnn::DataType outputType(cmdline.GetOutputType());
    const unsigned int newWidth  = cmdline.GetNewWidth();
    const unsigned int newHeight = cmdline.GetNewHeight();
    const unsigned int batchSize = 1;
    const armnn::DataLayout outputLayout(cmdline.GetLayout());

    std::vector<armnnUtils::TContainer> imageDataContainers;
    const NormalizationParameters& normParams = GetNormalizationParameters(modelFormat, outputType);
    try
    {
        switch (outputType)
        {
            case armnn::DataType::Signed32:
                imageDataContainers.push_back(PrepareImageTensor<int>(
                    imagePath, newWidth, newHeight, normParams, batchSize, outputLayout));
                break;
            case armnn::DataType::QAsymmU8:
                imageDataContainers.push_back(PrepareImageTensor<uint8_t>(
                    imagePath, newWidth, newHeight, normParams, batchSize, outputLayout));
                break;
            case armnn::DataType::QAsymmS8:
                imageDataContainers.push_back(PrepareImageTensor<int8_t>(
                        imagePath, newWidth, newHeight, normParams, batchSize, outputLayout));
                break;
            case armnn::DataType::Float32:
            default:
                imageDataContainers.push_back(PrepareImageTensor<float>(
                    imagePath, newWidth, newHeight, normParams, batchSize, outputLayout));
                break;
        }
    }
    catch (const InferenceTestImageException& e)
    {
        ARMNN_LOG(fatal) << "Failed to load image file " << imagePath << " with error: " << e.what();
        return -1;
    }

    std::ofstream imageTensorFile;
    imageTensorFile.open(outputPath, std::ofstream::out);
    if (imageTensorFile.is_open())
    {
        mapbox::util::apply_visitor(
            [&imageTensorFile](auto&& imageData){ WriteImageTensorImpl(imageData,imageTensorFile); },
            imageDataContainers[0]
            );

        if (!imageTensorFile)
        {
            ARMNN_LOG(fatal) << "Failed to write to output file" << outputPath;
            imageTensorFile.close();
            return -1;
        }
        imageTensorFile.close();
    }
    else
    {
        ARMNN_LOG(fatal) << "Failed to open output file" << outputPath;
        return -1;
    }

    return 0;
}
