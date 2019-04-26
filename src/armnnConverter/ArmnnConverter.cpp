//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <armnn/ArmNN.hpp>

#if defined(ARMNN_CAFFE_PARSER)
#include <armnnCaffeParser/ICaffeParser.hpp>
#endif
#if defined(ARMNN_ONNX_PARSER)
#include <armnnOnnxParser/IOnnxParser.hpp>
#endif
#if defined(ARMNN_SERIALIZER)
#include <armnnSerializer/ISerializer.hpp>
#endif
#if defined(ARMNN_TF_PARSER)
#include <armnnTfParser/ITfParser.hpp>
#endif
#if defined(ARMNN_TF_LITE_PARSER)
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#endif

#include <Logging.hpp>
#include <HeapProfiling.hpp>

#include <boost/format.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/program_options.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>

namespace
{

namespace po = boost::program_options;

armnn::TensorShape ParseTensorShape(std::istream& stream)
{
    std::vector<unsigned int> result;
    std::string line;

    while (std::getline(stream, line))
    {
        std::vector<std::string> tokens;
        try
        {
            // Coverity fix: boost::split() may throw an exception of type boost::bad_function_call.
            boost::split(tokens, line, boost::algorithm::is_any_of(","), boost::token_compress_on);
        }
        catch (const std::exception& e)
        {
            BOOST_LOG_TRIVIAL(error) << "An error occurred when splitting tokens: " << e.what();
            continue;
        }
        for (const std::string& token : tokens)
        {
            if (!token.empty())
            {
                try
                {
                    result.push_back(boost::numeric_cast<unsigned int>(std::stoi((token))));
                }
                catch (const std::exception&)
                {
                    BOOST_LOG_TRIVIAL(error) << "'" << token << "' is not a valid number. It has been ignored.";
                }
            }
        }
    }

    return armnn::TensorShape(boost::numeric_cast<unsigned int>(result.size()), result.data());
}

bool CheckOption(const po::variables_map& vm,
                 const char* option)
{
    if (option == nullptr)
    {
        return false;
    }

    // Check whether 'option' is provided.
    return vm.find(option) != vm.end();
}

void CheckOptionDependency(const po::variables_map& vm,
                           const char* option,
                           const char* required)
{
    if (option == nullptr || required == nullptr)
    {
        throw po::error("Invalid option to check dependency for");
    }

    // Check that if 'option' is provided, 'required' is also provided.
    if (CheckOption(vm, option) && !vm[option].defaulted())
    {
        if (CheckOption(vm, required) == 0 || vm[required].defaulted())
        {
            throw po::error(std::string("Option '") + option + "' requires option '" + required + "'.");
        }
    }
}

void CheckOptionDependencies(const po::variables_map& vm)
{
    CheckOptionDependency(vm, "model-path", "model-format");
    CheckOptionDependency(vm, "model-path", "input-name");
    CheckOptionDependency(vm, "model-path", "output-name");
    CheckOptionDependency(vm, "input-tensor-shape", "model-path");
}

int ParseCommandLineArgs(int argc, const char* argv[],
                         std::string& modelFormat,
                         std::string& modelPath,
                         std::vector<std::string>& inputNames,
                         std::vector<std::string>& inputTensorShapeStrs,
                         std::vector<std::string>& outputNames,
                         std::string& outputPath, bool& isModelBinary)
{
    po::options_description desc("Options");

    desc.add_options()
        ("help", "Display usage information")
        ("model-format,f", po::value(&modelFormat)->required(),"Format of the model file"
#if defined(ARMNN_CAFFE_PARSER)
         ", caffe-binary, caffe-text"
#endif
#if defined(ARMNN_ONNX_PARSER)
         ", onnx-binary, onnx-text"
#endif
#if defined(ARMNN_TF_PARSER)
         ", tensorflow-binary, tensorflow-text"
#endif
#if defined(ARMNN_TF_LITE_PARSER)
         ", tflite-binary"
#endif
         ".")
        ("model-path,m", po::value(&modelPath)->required(), "Path to model file.")
        ("input-name,i", po::value<std::vector<std::string>>()->multitoken(),
         "Identifier of the input tensors in the network, separated by whitespace.")
        ("input-tensor-shape,s", po::value<std::vector<std::string>>()->multitoken(),
         "The shape of the input tensor in the network as a flat array of integers, separated by comma."
         " Multiple shapes are separated by whitespace."
         " This parameter is optional, depending on the network.")
        ("output-name,o", po::value<std::vector<std::string>>()->multitoken(),
         "Identifier of the output tensor in the network.")
        ("output-path,p", po::value(&outputPath)->required(), "Path to serialize the network to.");

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (CheckOption(vm, "help") || argc <= 1)
        {
            std::cout << "Convert a neural network model from provided file to ArmNN format." << std::endl;
            std::cout << std::endl;
            std::cout << desc << std::endl;
            exit(EXIT_SUCCESS);
        }
        po::notify(vm);
    }
    catch (const po::error& e)
    {
        std::cerr << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
        return EXIT_FAILURE;
    }

    try
    {
        CheckOptionDependencies(vm);
    }
    catch (const po::error& e)
    {
        std::cerr << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
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
        BOOST_LOG_TRIVIAL(fatal) << "Unknown model format: '" << modelFormat << "'. Please include 'binary' or 'text'";
        return EXIT_FAILURE;
    }

    if (!vm["input-tensor-shape"].empty())
    {
        inputTensorShapeStrs = vm["input-tensor-shape"].as<std::vector<std::string>>();
    }

    inputNames = vm["input-name"].as<std::vector<std::string>>();
    outputNames = vm["output-name"].as<std::vector<std::string>>();

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
                throw armnn::Exception(boost::str(boost::format(
                    "Not every input has its tensor shape specified: expected=%1%, got=%2%")
                    % numInputBindings % numInputShapes));
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
                throw armnn::Exception(boost::str(boost::format(
                    "Not every input has its tensor shape specified: expected=%1%, got=%2%")
                    % numInputBindings % numInputShapes));
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
                throw armnn::Exception(boost::str(boost::format(
                    "Not every input has its tensor shape specified: expected=%1%, got=%2%")
                    % numInputBindings % numInputShapes));
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

int main(int argc, const char* argv[])
{

#if (!defined(ARMNN_CAFFE_PARSER)     \
       && !defined(ARMNN_ONNX_PARSER) \
       && !defined(ARMNN_TF_PARSER)   \
       && !defined(ARMNN_TF_LITE_PARSER))
    BOOST_LOG_TRIVIAL(fatal) << "Not built with any of the supported parsers, Caffe, Onnx, Tensorflow, or TfLite.";
    return EXIT_FAILURE;
#endif

#if !defined(ARMNN_SERIALIZER)
    BOOST_LOG_TRIVIAL(fatal) << "Not built with Serializer support.";
    return EXIT_FAILURE;
#endif

#ifdef NDEBUG
    armnn::LogSeverity level = armnn::LogSeverity::Info;
#else
    armnn::LogSeverity level = armnn::LogSeverity::Debug;
#endif

    armnn::ConfigureLogging(true, true, level);
    armnnUtils::ConfigureLogging(boost::log::core::get().get(), true, true, level);

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
                BOOST_LOG_TRIVIAL(fatal) << "Cannot create tensor shape: " << e.what();
                return EXIT_FAILURE;
            }
        }
    }

    ArmnnConverter converter(modelPath, inputNames, inputTensorShapes, outputNames, outputPath, isModelBinary);

    if (modelFormat.find("caffe") != std::string::npos)
    {
#if defined(ARMNN_CAFFE_PARSER)
        if (!converter.CreateNetwork<armnnCaffeParser::ICaffeParser>())
        {
            BOOST_LOG_TRIVIAL(fatal) << "Failed to load model from file";
            return EXIT_FAILURE;
        }
#else
        BOOST_LOG_TRIVIAL(fatal) << "Not built with Caffe parser support.";
        return EXIT_FAILURE;
#endif
    }
    else if (modelFormat.find("onnx") != std::string::npos)
    {
#if defined(ARMNN_ONNX_PARSER)
        if (!converter.CreateNetwork<armnnOnnxParser::IOnnxParser>())
        {
            BOOST_LOG_TRIVIAL(fatal) << "Failed to load model from file";
            return EXIT_FAILURE;
        }
#else
        BOOST_LOG_TRIVIAL(fatal) << "Not built with Onnx parser support.";
        return EXIT_FAILURE;
#endif
    }
    else if (modelFormat.find("tensorflow") != std::string::npos)
    {
#if defined(ARMNN_TF_PARSER)
        if (!converter.CreateNetwork<armnnTfParser::ITfParser>())
        {
            BOOST_LOG_TRIVIAL(fatal) << "Failed to load model from file";
            return EXIT_FAILURE;
        }
#else
        BOOST_LOG_TRIVIAL(fatal) << "Not built with Tensorflow parser support.";
        return EXIT_FAILURE;
#endif
    }
    else if (modelFormat.find("tflite") != std::string::npos)
    {
#if defined(ARMNN_TF_LITE_PARSER)
        if (!isModelBinary)
        {
            BOOST_LOG_TRIVIAL(fatal) << "Unknown model format: '" << modelFormat << "'. Only 'binary' format supported \
              for tflite files";
            return EXIT_FAILURE;
        }

        if (!converter.CreateNetwork<armnnTfLiteParser::ITfLiteParser>())
        {
            BOOST_LOG_TRIVIAL(fatal) << "Failed to load model from file";
            return EXIT_FAILURE;
        }
#else
        BOOST_LOG_TRIVIAL(fatal) << "Not built with TfLite parser support.";
        return EXIT_FAILURE;
#endif
    }
    else
    {
        BOOST_LOG_TRIVIAL(fatal) << "Unknown model format: '" << modelFormat << "'";
        return EXIT_FAILURE;
    }

    if (!converter.Serialize())
    {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to serialize model";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
