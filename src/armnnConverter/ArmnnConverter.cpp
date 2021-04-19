//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <armnn/Logging.hpp>

#if defined(ARMNN_ONNX_PARSER)
#include <armnnOnnxParser/IOnnxParser.hpp>
#endif
#if defined(ARMNN_SERIALIZER)
#include <armnnSerializer/ISerializer.hpp>
#endif
#if defined(ARMNN_TF_LITE_PARSER)
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#endif

#include <HeapProfiling.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/StringUtils.hpp>

/*
 * Historically we use the ',' character to separate dimensions in a tensor shape. However, cxxopts will read this
 * an an array of values which is fine until we have multiple tensors specified. This lumps the values of all shapes
 * together in a single array and we cannot break it up again. We'll change the vector delimiter to a '.'. We do this
 * as close as possible to the usage of cxxopts to avoid polluting other possible uses.
 */
#define CXXOPTS_VECTOR_DELIMITER '.'
#include <cxxopts/cxxopts.hpp>

#include <fmt/format.h>

#include <cstdlib>
#include <fstream>
#include <iostream>

namespace
{

armnn::TensorShape ParseTensorShape(std::istream& stream)
{
    std::vector<unsigned int> result;
    std::string line;

    while (std::getline(stream, line))
    {
        std::vector<std::string> tokens = armnn::stringUtils::StringTokenizer(line, ",");
        for (const std::string& token : tokens)
        {
            if (!token.empty())
            {
                try
                {
                    result.push_back(armnn::numeric_cast<unsigned int>(std::stoi((token))));
                }
                catch (const std::exception&)
                {
                    ARMNN_LOG(error) << "'" << token << "' is not a valid number. It has been ignored.";
                }
            }
        }
    }

    return armnn::TensorShape(armnn::numeric_cast<unsigned int>(result.size()), result.data());
}

int ParseCommandLineArgs(int argc, char* argv[],
                         std::string& modelFormat,
                         std::string& modelPath,
                         std::vector<std::string>& inputNames,
                         std::vector<std::string>& inputTensorShapeStrs,
                         std::vector<std::string>& outputNames,
                         std::string& outputPath, bool& isModelBinary)
{
    cxxopts::Options options("ArmNNConverter", "Convert a neural network model from provided file to ArmNN format.");
    try
    {
        std::string modelFormatDescription("Format of the model file");
#if defined(ARMNN_ONNX_PARSER)
        modelFormatDescription += ", onnx-binary, onnx-text";
#endif
#if defined(ARMNN_TF_PARSER)
        modelFormatDescription += ", tensorflow-binary, tensorflow-text";
#endif
#if defined(ARMNN_TF_LITE_PARSER)
        modelFormatDescription += ", tflite-binary";
#endif
        modelFormatDescription += ".";
        options.add_options()
            ("help", "Display usage information")
            ("f,model-format", modelFormatDescription, cxxopts::value<std::string>(modelFormat))
            ("m,model-path", "Path to model file.", cxxopts::value<std::string>(modelPath))

            ("i,input-name", "Identifier of the input tensors in the network. "
                             "Each input must be specified separately.",
                             cxxopts::value<std::vector<std::string>>(inputNames))
            ("s,input-tensor-shape",
                             "The shape of the input tensor in the network as a flat array of integers, "
                             "separated by comma. Each input shape must be specified separately after the input name. "
                             "This parameter is optional, depending on the network.",
                             cxxopts::value<std::vector<std::string>>(inputTensorShapeStrs))

            ("o,output-name", "Identifier of the output tensor in the network.",
                              cxxopts::value<std::vector<std::string>>(outputNames))
            ("p,output-path",
                         "Path to serialize the network to.", cxxopts::value<std::string>(outputPath));
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl << options.help() << std::endl;
        return EXIT_FAILURE;
    }
    try
    {
        cxxopts::ParseResult result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cerr << options.help()  << std::endl;
            return EXIT_SUCCESS;
        }
        // Check for mandatory single options.
        std::string mandatorySingleParameters[] = { "model-format", "model-path", "output-name", "output-path" };
        bool somethingsMissing = false;
        for (auto param : mandatorySingleParameters)
        {
            if (result.count(param) != 1)
            {
                std::cerr << "Parameter \'--" << param << "\' is required but missing." << std::endl;
                somethingsMissing = true;
            }
        }
        // Check at least one "input-name" option.
        if (result.count("input-name") == 0)
        {
            std::cerr << "Parameter \'--" << "input-name" << "\' must be specified at least once." << std::endl;
            somethingsMissing = true;
        }
        // If input-tensor-shape is specified then there must be a 1:1 match with input-name.
        if (result.count("input-tensor-shape") > 0)
        {
            if (result.count("input-tensor-shape") != result.count("input-name"))
            {
                std::cerr << "When specifying \'input-tensor-shape\' a matching number of \'input-name\' parameters "
                             "must be specified." << std::endl;
                somethingsMissing = true;
            }
        }

        if (somethingsMissing)
        {
            std::cerr << options.help()  << std::endl;
            return EXIT_FAILURE;
        }
    }
    catch (const cxxopts::OptionException& e)
    {
        std::cerr << e.what() << std::endl << std::endl;
        return EXIT_FAILURE;
    }

    if (modelFormat.find("bin") != std::string::npos)
    {
        isModelBinary = true;
    }
    else if (modelFormat.find("text") != std::string::npos)
    {
        isModelBinary = false;
    }
    else
    {
        ARMNN_LOG(fatal) << "Unknown model format: '" << modelFormat << "'. Please include 'binary' or 'text'";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

template<typename T>
struct ParserType
{
    typedef T parserType;
};

class ArmnnConverter
{
public:
    ArmnnConverter(const std::string& modelPath,
                   const std::vector<std::string>& inputNames,
                   const std::vector<armnn::TensorShape>& inputShapes,
                   const std::vector<std::string>& outputNames,
                   const std::string& outputPath,
                   bool isModelBinary)
    : m_NetworkPtr(armnn::INetworkPtr(nullptr, [](armnn::INetwork *){})),
    m_ModelPath(modelPath),
    m_InputNames(inputNames),
    m_InputShapes(inputShapes),
    m_OutputNames(outputNames),
    m_OutputPath(outputPath),
    m_IsModelBinary(isModelBinary) {}

    bool Serialize()
    {
        if (m_NetworkPtr.get() == nullptr)
        {
            return false;
        }

        auto serializer(armnnSerializer::ISerializer::Create());

        serializer->Serialize(*m_NetworkPtr);

        std::ofstream file(m_OutputPath, std::ios::out | std::ios::binary);

        bool retVal = serializer->SaveSerializedToStream(file);

        return retVal;
    }

    template <typename IParser>
    bool CreateNetwork ()
    {
        return CreateNetwork (ParserType<IParser>());
    }

private:
    armnn::INetworkPtr              m_NetworkPtr;
    std::string                     m_ModelPath;
    std::vector<std::string>        m_InputNames;
    std::vector<armnn::TensorShape> m_InputShapes;
    std::vector<std::string>        m_OutputNames;
    std::string                     m_OutputPath;
    bool                            m_IsModelBinary;

    template <typename IParser>
    bool CreateNetwork (ParserType<IParser>)
    {
        // Create a network from a file on disk
        auto parser(IParser::Create());

        std::map<std::string, armnn::TensorShape> inputShapes;
        if (!m_InputShapes.empty())
        {
            const size_t numInputShapes   = m_InputShapes.size();
            const size_t numInputBindings = m_InputNames.size();
            if (numInputShapes < numInputBindings)
            {
                throw armnn::Exception(fmt::format(
                   "Not every input has its tensor shape specified: expected={0}, got={1}",
                   numInputBindings, numInputShapes));
            }

            for (size_t i = 0; i < numInputShapes; i++)
            {
                inputShapes[m_InputNames[i]] = m_InputShapes[i];
            }
        }

        {
            ARMNN_SCOPED_HEAP_PROFILING("Parsing");
            m_NetworkPtr = (m_IsModelBinary ?
                parser->CreateNetworkFromBinaryFile(m_ModelPath.c_str(), inputShapes, m_OutputNames) :
                parser->CreateNetworkFromTextFile(m_ModelPath.c_str(), inputShapes, m_OutputNames));
        }

        return m_NetworkPtr.get() != nullptr;
    }

#if defined(ARMNN_TF_LITE_PARSER)
    bool CreateNetwork (ParserType<armnnTfLiteParser::ITfLiteParser>)
    {
        // Create a network from a file on disk
        auto parser(armnnTfLiteParser::ITfLiteParser::Create());

        if (!m_InputShapes.empty())
        {
            const size_t numInputShapes   = m_InputShapes.size();
            const size_t numInputBindings = m_InputNames.size();
            if (numInputShapes < numInputBindings)
            {
                throw armnn::Exception(fmt::format(
                   "Not every input has its tensor shape specified: expected={0}, got={1}",
                   numInputBindings, numInputShapes));
            }
        }

        {
            ARMNN_SCOPED_HEAP_PROFILING("Parsing");
            m_NetworkPtr = parser->CreateNetworkFromBinaryFile(m_ModelPath.c_str());
        }

        return m_NetworkPtr.get() != nullptr;
    }
#endif

#if defined(ARMNN_ONNX_PARSER)
    bool CreateNetwork (ParserType<armnnOnnxParser::IOnnxParser>)
    {
        // Create a network from a file on disk
        auto parser(armnnOnnxParser::IOnnxParser::Create());

        if (!m_InputShapes.empty())
        {
            const size_t numInputShapes   = m_InputShapes.size();
            const size_t numInputBindings = m_InputNames.size();
            if (numInputShapes < numInputBindings)
            {
                throw armnn::Exception(fmt::format(
                   "Not every input has its tensor shape specified: expected={0}, got={1}",
                   numInputBindings, numInputShapes));
            }
        }

        {
            ARMNN_SCOPED_HEAP_PROFILING("Parsing");
            m_NetworkPtr = (m_IsModelBinary ?
                parser->CreateNetworkFromBinaryFile(m_ModelPath.c_str()) :
                parser->CreateNetworkFromTextFile(m_ModelPath.c_str()));
        }

        return m_NetworkPtr.get() != nullptr;
    }
#endif

};

} // anonymous namespace

int main(int argc, char* argv[])
{

#if (!defined(ARMNN_ONNX_PARSER) \
       && !defined(ARMNN_TF_PARSER)   \
       && !defined(ARMNN_TF_LITE_PARSER))
    ARMNN_LOG(fatal) << "Not built with any of the supported parsers Onnx, Tensorflow, or TfLite.";
    return EXIT_FAILURE;
#endif

#if !defined(ARMNN_SERIALIZER)
    ARMNN_LOG(fatal) << "Not built with Serializer support.";
    return EXIT_FAILURE;
#endif

#ifdef NDEBUG
    armnn::LogSeverity level = armnn::LogSeverity::Info;
#else
    armnn::LogSeverity level = armnn::LogSeverity::Debug;
#endif

    armnn::ConfigureLogging(true, true, level);

    std::string modelFormat;
    std::string modelPath;

    std::vector<std::string> inputNames;
    std::vector<std::string> inputTensorShapeStrs;
    std::vector<armnn::TensorShape> inputTensorShapes;

    std::vector<std::string> outputNames;
    std::string outputPath;

    bool isModelBinary = true;

    if (ParseCommandLineArgs(
        argc, argv, modelFormat, modelPath, inputNames, inputTensorShapeStrs, outputNames, outputPath, isModelBinary)
        != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }

    for (const std::string& shapeStr : inputTensorShapeStrs)
    {
        if (!shapeStr.empty())
        {
            std::stringstream ss(shapeStr);

            try
            {
                armnn::TensorShape shape = ParseTensorShape(ss);
                inputTensorShapes.push_back(shape);
            }
            catch (const armnn::InvalidArgumentException& e)
            {
                ARMNN_LOG(fatal) << "Cannot create tensor shape: " << e.what();
                return EXIT_FAILURE;
            }
        }
    }

    ArmnnConverter converter(modelPath, inputNames, inputTensorShapes, outputNames, outputPath, isModelBinary);

    try
    {
        if (modelFormat.find("onnx") != std::string::npos)
        {
#if defined(ARMNN_ONNX_PARSER)
            if (!converter.CreateNetwork<armnnOnnxParser::IOnnxParser>())
            {
                ARMNN_LOG(fatal) << "Failed to load model from file";
                return EXIT_FAILURE;
            }
#else
            ARMNN_LOG(fatal) << "Not built with Onnx parser support.";
            return EXIT_FAILURE;
#endif
        }
        else if (modelFormat.find("tflite") != std::string::npos)
        {
#if defined(ARMNN_TF_LITE_PARSER)
            if (!isModelBinary)
            {
                ARMNN_LOG(fatal) << "Unknown model format: '" << modelFormat << "'. Only 'binary' format supported \
                  for tflite files";
                return EXIT_FAILURE;
            }

            if (!converter.CreateNetwork<armnnTfLiteParser::ITfLiteParser>())
            {
                ARMNN_LOG(fatal) << "Failed to load model from file";
                return EXIT_FAILURE;
            }
#else
            ARMNN_LOG(fatal) << "Not built with TfLite parser support.";
            return EXIT_FAILURE;
#endif
        }
        else
        {
            ARMNN_LOG(fatal) << "Unknown model format: '" << modelFormat << "'";
            return EXIT_FAILURE;
        }
    }
    catch(armnn::Exception& e)
    {
        ARMNN_LOG(fatal) << "Failed to load model from file: " << e.what();
        return EXIT_FAILURE;
    }

    if (!converter.Serialize())
    {
        ARMNN_LOG(fatal) << "Failed to serialize model";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
