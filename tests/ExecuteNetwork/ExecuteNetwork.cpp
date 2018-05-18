//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "armnn/ArmNN.hpp"
#if defined(ARMNN_CAFFE_PARSER)
#include "armnnCaffeParser/ICaffeParser.hpp"
#endif
#if defined(ARMNN_TF_PARSER)
#include "armnnTfParser/ITfParser.hpp"
#endif
#include "Logging.hpp"
#include "../InferenceTest.hpp"

#include <boost/program_options.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <iostream>
#include <fstream>

namespace
{

template<typename T, typename TParseElementFunc>
std::vector<T> ParseArrayImpl(std::istream& stream, TParseElementFunc parseElementFunc)
{
    std::vector<T> result;
    // Process line-by-line
    std::string line;
    while (std::getline(stream, line))
    {
        std::vector<std::string> tokens;
        try
        {
            // Coverity fix: boost::split() may throw an exception of type boost::bad_function_call.
            boost::split(tokens, line, boost::algorithm::is_any_of("\t ,;:"), boost::token_compress_on);
        }
        catch (const std::exception& e)
        {
            BOOST_LOG_TRIVIAL(error) << "An error occurred when splitting tokens: " << e.what();
            continue;
        }
        for (const std::string& token : tokens)
        {
            if (!token.empty()) // See https://stackoverflow.com/questions/10437406/
            {
                try
                {
                    result.push_back(parseElementFunc(token));
                }
                catch (const std::exception&)
                {
                    BOOST_LOG_TRIVIAL(error) << "'" << token << "' is not a valid number. It has been ignored.";
                }
            }
        }
    }

    return result;
}

}

template<typename T>
std::vector<T> ParseArray(std::istream& stream);

template<>
std::vector<float> ParseArray(std::istream& stream)
{
    return ParseArrayImpl<float>(stream, [](const std::string& s) { return std::stof(s); });
}

template<>
std::vector<unsigned int> ParseArray(std::istream& stream)
{
    return ParseArrayImpl<unsigned int>(stream,
        [](const std::string& s) { return boost::numeric_cast<unsigned int>(std::stoi(s)); });
}

void PrintArray(const std::vector<float>& v)
{
    for (size_t i = 0; i < v.size(); i++)
    {
        printf("%f ", v[i]);
    }
    printf("\n");
}

template<typename TParser, typename TDataType>
int MainImpl(const char* modelPath, bool isModelBinary, armnn::Compute computeDevice,
    const char* inputName, const armnn::TensorShape* inputTensorShape, const char* inputTensorDataFilePath,
    const char* outputName)
{
    // Load input tensor
    std::vector<TDataType> input;
    {
        std::ifstream inputTensorFile(inputTensorDataFilePath);
        if (!inputTensorFile.good())
        {
            BOOST_LOG_TRIVIAL(fatal) << "Failed to load input tensor data file from " << inputTensorDataFilePath;
            return 1;
        }
        input = ParseArray<TDataType>(inputTensorFile);
    }

    try
    {
        // Create an InferenceModel, which will parse the model and load it into an IRuntime
        typename InferenceModel<TParser, TDataType>::Params params;
        params.m_ModelPath = modelPath;
        params.m_IsModelBinary = isModelBinary;
        params.m_ComputeDevice = computeDevice;
        params.m_InputBinding = inputName;
        params.m_InputTensorShape = inputTensorShape;
        params.m_OutputBinding = outputName;
        InferenceModel<TParser, TDataType> model(params);

        // Execute the model
        std::vector<TDataType> output(model.GetOutputSize());
        model.Run(input, output);

        // Print the output tensor
        PrintArray(output);
    }
    catch (armnn::Exception const& e)
    {
        BOOST_LOG_TRIVIAL(fatal) << "Armnn Error: " << e.what();
        return 1;
    }

    return 0;
}

int main(int argc, char* argv[])
{
    // Configure logging for both the ARMNN library and this test program
#ifdef NDEBUG
    armnn::LogSeverity level = armnn::LogSeverity::Info;
#else
    armnn::LogSeverity level = armnn::LogSeverity::Debug;
#endif
    armnn::ConfigureLogging(true, true, level);
    armnnUtils::ConfigureLogging(boost::log::core::get().get(), true, true, level);

    // Configure boost::program_options for command-line parsing
    namespace po = boost::program_options;

    std::string modelFormat;
    std::string modelPath;
    std::string inputName;
    std::string inputTensorShapeStr;
    std::string inputTensorDataFilePath;
    std::string outputName;
    armnn::Compute computeDevice;

    po::options_description desc("Options");
    try
    {
        desc.add_options()
            ("help", "Display usage information")
            ("model-format,f", po::value(&modelFormat)->required(),
                "caffe-binary, caffe-text, tensorflow-binary or tensorflow-text.")
            ("model-path,m", po::value(&modelPath)->required(), "Path to model file, e.g. .caffemodel, .prototxt")
            ("compute,c", po::value<armnn::Compute>(&computeDevice)->required(),
                "Which device to run layers on by default. Possible choices: CpuAcc, CpuRef, GpuAcc")
            ("input-name,i", po::value(&inputName)->required(), "Identifier of the input tensor in the network.")
            ("input-tensor-shape,s", po::value(&inputTensorShapeStr),
                "The shape of the input tensor in the network as a flat array of integers separated by whitespace. "
                "This parameter is optional, depending on the network.")
            ("input-tensor-data,d", po::value(&inputTensorDataFilePath)->required(),
             "Path to a file containing the input data as a flat array separated by whitespace.")
            ("output-name,o", po::value(&outputName)->required(), "Identifier of the output tensor in the network.");
    }
    catch (const std::exception& e)
    {
        // Coverity points out that default_value(...) can throw a bad_lexical_cast,
        // and that desc.add_options() can throw boost::io::too_few_args.
        // They really won't in any of these cases.
        BOOST_ASSERT_MSG(false, "Caught unexpected exception");
        BOOST_LOG_TRIVIAL(fatal) << "Fatal internal error: " << e.what();
        return 1;
    }

    // Parse the command-line
    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help") || argc <= 1)
        {
            std::cout << "Executes a neural network model using the provided input tensor. " << std::endl;
            std::cout << "Prints the resulting output tensor." << std::endl;
            std::cout << std::endl;
            std::cout << desc << std::endl;
            return 1;
        }

        po::notify(vm);
    }
    catch (po::error& e)
    {
        std::cerr << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }

    // Parse model binary flag from the model-format string we got from the command-line
    bool isModelBinary;
    if (modelFormat.find("bin") != std::string::npos)
    {
        isModelBinary = true;
    }
    else if (modelFormat.find("txt") != std::string::npos || modelFormat.find("text") != std::string::npos)
    {
        isModelBinary = false;
    }
    else
    {
        BOOST_LOG_TRIVIAL(fatal) << "Unknown model format: '" << modelFormat << "'. Please include 'binary' or 'text'";
        return 1;
    }

    // Parse input tensor shape from the string we got from the command-line.
    std::unique_ptr<armnn::TensorShape> inputTensorShape;
    if (!inputTensorShapeStr.empty())
    {
        std::stringstream ss(inputTensorShapeStr);
        std::vector<unsigned int> dims = ParseArray<unsigned int>(ss);

        try
        {
            // Coverity fix: An exception of type armnn::InvalidArgumentException is thrown and never caught.
            inputTensorShape = std::make_unique<armnn::TensorShape>(dims.size(), dims.data());
        }
        catch (const armnn::InvalidArgumentException& e)
        {
            BOOST_LOG_TRIVIAL(fatal) << "Cannot create tensor shape: " << e.what();
            return 1;
        }
    }

    // Forward to implementation based on the parser type
    if (modelFormat.find("caffe") != std::string::npos)
    {
#if defined(ARMNN_CAFFE_PARSER)
        return MainImpl<armnnCaffeParser::ICaffeParser, float>(modelPath.c_str(), isModelBinary, computeDevice,
            inputName.c_str(), inputTensorShape.get(), inputTensorDataFilePath.c_str(), outputName.c_str());
#else
        BOOST_LOG_TRIVIAL(fatal) << "Not built with Caffe parser support.";
        return 1;
#endif
    }
    else if (modelFormat.find("tensorflow") != std::string::npos)
    {
#if defined(ARMNN_TF_PARSER)
        return MainImpl<armnnTfParser::ITfParser, float>(modelPath.c_str(), isModelBinary, computeDevice,
            inputName.c_str(), inputTensorShape.get(), inputTensorDataFilePath.c_str(), outputName.c_str());
#else
        BOOST_LOG_TRIVIAL(fatal) << "Not built with Tensorflow parser support.";
        return 1;
#endif
    }
    else
    {
        BOOST_LOG_TRIVIAL(fatal) << "Unknown model format: '" << modelFormat <<
            "'. Please include 'caffe' or 'tensorflow'";
        return 1;
    }
}
