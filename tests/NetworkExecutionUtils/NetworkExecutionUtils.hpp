//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <armnn/ArmNN.hpp>
#include <armnn/TypesUtils.hpp>

#if defined(ARMNN_SERIALIZER)
#include "armnnDeserializer/IDeserializer.hpp"
#endif
#if defined(ARMNN_CAFFE_PARSER)
#include "armnnCaffeParser/ICaffeParser.hpp"
#endif
#if defined(ARMNN_TF_PARSER)
#include "armnnTfParser/ITfParser.hpp"
#endif
#if defined(ARMNN_TF_LITE_PARSER)
#include "armnnTfLiteParser/ITfLiteParser.hpp"
#endif
#if defined(ARMNN_ONNX_PARSER)
#include "armnnOnnxParser/IOnnxParser.hpp"
#endif
#include "CsvReader.hpp"
#include "../InferenceTest.hpp"

#include <Logging.hpp>
#include <Profiling.hpp>

#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/program_options.hpp>
#include <boost/variant.hpp>

#include <iostream>
#include <fstream>
#include <functional>
#include <future>
#include <algorithm>
#include <iterator>

namespace
{

// Configure boost::program_options for command-line parsing and validation.
namespace po = boost::program_options;

template<typename T, typename TParseElementFunc>
std::vector<T> ParseArrayImpl(std::istream& stream, TParseElementFunc parseElementFunc, const char * chars = "\t ,:")
{
    std::vector<T> result;
    // Processes line-by-line.
    std::string line;
    while (std::getline(stream, line))
    {
        std::vector<std::string> tokens;
        try
        {
            // Coverity fix: boost::split() may throw an exception of type boost::bad_function_call.
            boost::split(tokens, line, boost::algorithm::is_any_of(chars), boost::token_compress_on);
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

bool CheckOption(const po::variables_map& vm,
                 const char* option)
{
    // Check that the given option is valid.
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
    // Check that the given options are valid.
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
    CheckOptionDependency(vm, "model-path", "input-tensor-data");
    CheckOptionDependency(vm, "model-path", "output-name");
    CheckOptionDependency(vm, "input-tensor-shape", "model-path");
}

template<armnn::DataType NonQuantizedType>
auto ParseDataArray(std::istream & stream);

template<armnn::DataType QuantizedType>
auto ParseDataArray(std::istream& stream,
                    const float& quantizationScale,
                    const int32_t& quantizationOffset);

template<>
auto ParseDataArray<armnn::DataType::Float32>(std::istream & stream)
{
    return ParseArrayImpl<float>(stream, [](const std::string& s) { return std::stof(s); });
}

template<>
auto ParseDataArray<armnn::DataType::Signed32>(std::istream & stream)
{
    return ParseArrayImpl<int>(stream, [](const std::string & s) { return std::stoi(s); });
}

template<>
auto ParseDataArray<armnn::DataType::QuantisedAsymm8>(std::istream& stream)
{
    return ParseArrayImpl<uint8_t>(stream,
                                   [](const std::string& s) { return boost::numeric_cast<uint8_t>(std::stoi(s)); });
}

template<>
auto ParseDataArray<armnn::DataType::QuantisedAsymm8>(std::istream& stream,
                                                      const float& quantizationScale,
                                                      const int32_t& quantizationOffset)
{
    return ParseArrayImpl<uint8_t>(stream,
                                   [&quantizationScale, &quantizationOffset](const std::string & s)
                                   {
                                       return boost::numeric_cast<uint8_t>(
                                           armnn::Quantize<u_int8_t>(std::stof(s),
                                                                     quantizationScale,
                                                                     quantizationOffset));
                                   });
}
std::vector<unsigned int> ParseArray(std::istream& stream)
{
    return ParseArrayImpl<unsigned int>(stream,
        [](const std::string& s) { return boost::numeric_cast<unsigned int>(std::stoi(s)); });
}

std::vector<std::string> ParseStringList(const std::string & inputString, const char * delimiter)
{
    std::stringstream stream(inputString);
    return ParseArrayImpl<std::string>(stream, [](const std::string& s) { return boost::trim_copy(s); }, delimiter);
}

void RemoveDuplicateDevices(std::vector<armnn::BackendId>& computeDevices)
{
    // Mark the duplicate devices as 'Undefined'.
    for (auto i = computeDevices.begin(); i != computeDevices.end(); ++i)
    {
        for (auto j = std::next(i); j != computeDevices.end(); ++j)
        {
            if (*j == *i)
            {
                *j = armnn::Compute::Undefined;
            }
        }
    }

    // Remove 'Undefined' devices.
    computeDevices.erase(std::remove(computeDevices.begin(), computeDevices.end(), armnn::Compute::Undefined),
                         computeDevices.end());
}

struct TensorPrinter : public boost::static_visitor<>
{
    TensorPrinter(const std::string& binding, const armnn::TensorInfo& info)
        : m_OutputBinding(binding)
        , m_Scale(info.GetQuantizationScale())
        , m_Offset(info.GetQuantizationOffset())
    {}

    void operator()(const std::vector<float>& values)
    {
        ForEachValue(values, [](float value){
                printf("%f ", value);
            });
    }

    void operator()(const std::vector<uint8_t>& values)
    {
        auto& scale = m_Scale;
        auto& offset = m_Offset;
        ForEachValue(values, [&scale, &offset](uint8_t value)
            {
                printf("%f ", armnn::Dequantize(value, scale, offset));
            });
    }

    void operator()(const std::vector<int>& values)
    {
        ForEachValue(values, [](int value)
            {
                printf("%d ", value);
            });
    }

private:
    template<typename Container, typename Delegate>
    void ForEachValue(const Container& c, Delegate delegate)
    {
        std::cout << m_OutputBinding << ": ";
        for (const auto& value : c)
        {
            delegate(value);
        }
        printf("\n");
    }

    std::string m_OutputBinding;
    float m_Scale=0.0f;
    int m_Offset=0;
};


} // namespace

template<typename TParser, typename TDataType>
int MainImpl(const char* modelPath,
             bool isModelBinary,
             const std::vector<armnn::BackendId>& computeDevices,
             const std::string& dynamicBackendsPath,
             const std::vector<string>& inputNames,
             const std::vector<std::unique_ptr<armnn::TensorShape>>& inputTensorShapes,
             const std::vector<string>& inputTensorDataFilePaths,
             const std::vector<string>& inputTypes,
             bool quantizeInput,
             const std::vector<string>& outputTypes,
             const std::vector<string>& outputNames,
             bool enableProfiling,
             bool enableFp16TurboMode,
             const double& thresholdTime,
             const size_t subgraphId,
             const std::shared_ptr<armnn::IRuntime>& runtime = nullptr)
{
    using TContainer = boost::variant<std::vector<float>, std::vector<int>, std::vector<unsigned char>>;

    std::vector<TContainer> inputDataContainers;

    try
    {
        // Creates an InferenceModel, which will parse the model and load it into an IRuntime.
        typename InferenceModel<TParser, TDataType>::Params params;
        params.m_ModelPath = modelPath;
        params.m_IsModelBinary = isModelBinary;
        params.m_ComputeDevices = computeDevices;
        params.m_DynamicBackendsPath = dynamicBackendsPath;

        for(const std::string& inputName: inputNames)
        {
            params.m_InputBindings.push_back(inputName);
        }

        for(unsigned int i = 0; i < inputTensorShapes.size(); ++i)
        {
            params.m_InputShapes.push_back(*inputTensorShapes[i]);
        }

        for(const std::string& outputName: outputNames)
        {
            params.m_OutputBindings.push_back(outputName);
        }

        params.m_SubgraphId = subgraphId;
        params.m_EnableFp16TurboMode = enableFp16TurboMode;
        InferenceModel<TParser, TDataType> model(params, enableProfiling, dynamicBackendsPath, runtime);

        for(unsigned int i = 0; i < inputTensorDataFilePaths.size(); ++i)
        {
            std::ifstream inputTensorFile(inputTensorDataFilePaths[i]);

            if (inputTypes[i].compare("float") == 0)
            {
                if (quantizeInput)
                {
                    auto inputBinding = model.GetInputBindingInfo();
                    inputDataContainers.push_back(
                            ParseDataArray<armnn::DataType::QuantisedAsymm8>(inputTensorFile,
                                                         inputBinding.second.GetQuantizationScale(),
                                                         inputBinding.second.GetQuantizationOffset()));
                }
                else
                {
                    inputDataContainers.push_back(
                            ParseDataArray<armnn::DataType::Float32>(inputTensorFile));
                }
            }
            else if (inputTypes[i].compare("int") == 0)
            {
                inputDataContainers.push_back(
                    ParseDataArray<armnn::DataType::Signed32>(inputTensorFile));
            }
            else if (inputTypes[i].compare("qasymm8") == 0)
            {
                inputDataContainers.push_back(
                    ParseDataArray<armnn::DataType::QuantisedAsymm8>(inputTensorFile));
            }
            else
            {
                BOOST_LOG_TRIVIAL(fatal) << "Unsupported tensor data type \"" << inputTypes[i] << "\". ";
                return EXIT_FAILURE;
            }

            inputTensorFile.close();
        }

        const size_t numOutputs = params.m_OutputBindings.size();
        std::vector<TContainer> outputDataContainers;

        for (unsigned int i = 0; i < numOutputs; ++i)
        {
            if (outputTypes[i].compare("float") == 0)
            {
                outputDataContainers.push_back(std::vector<float>(model.GetOutputSize(i)));
            }
            else if (outputTypes[i].compare("int") == 0)
            {
                outputDataContainers.push_back(std::vector<int>(model.GetOutputSize(i)));
            }
            else if (outputTypes[i].compare("qasymm8") == 0)
            {
                outputDataContainers.push_back(std::vector<uint8_t>(model.GetOutputSize(i)));
            }
            else
            {
                BOOST_LOG_TRIVIAL(fatal) << "Unsupported tensor data type \"" << outputTypes[i] << "\". ";
                return EXIT_FAILURE;
            }
        }

        // model.Run returns the inference time elapsed in EnqueueWorkload (in milliseconds)
        auto inference_duration = model.Run(inputDataContainers, outputDataContainers);

        // Print output tensors
        const auto& infosOut = model.GetOutputBindingInfos();
        for (size_t i = 0; i < numOutputs; i++)
        {
            const armnn::TensorInfo& infoOut = infosOut[i].second;
            TensorPrinter printer(params.m_OutputBindings[i], infoOut);
            boost::apply_visitor(printer, outputDataContainers[i]);
        }

        BOOST_LOG_TRIVIAL(info) << "\nInference time: " << std::setprecision(2)
                                << std::fixed << inference_duration.count() << " ms";

        // If thresholdTime == 0.0 (default), then it hasn't been supplied at command line
        if (thresholdTime != 0.0)
        {
            BOOST_LOG_TRIVIAL(info) << "Threshold time: " << std::setprecision(2)
                                    << std::fixed << thresholdTime << " ms";
            auto thresholdMinusInference = thresholdTime - inference_duration.count();
            BOOST_LOG_TRIVIAL(info) << "Threshold time - Inference time: " << std::setprecision(2)
                                    << std::fixed << thresholdMinusInference << " ms" << "\n";

            if (thresholdMinusInference < 0)
            {
                BOOST_LOG_TRIVIAL(fatal) << "Elapsed inference time is greater than provided threshold time.\n";
                return EXIT_FAILURE;
            }
        }


    }
    catch (armnn::Exception const& e)
    {
        BOOST_LOG_TRIVIAL(fatal) << "Armnn Error: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// This will run a test
int RunTest(const std::string& format,
            const std::string& inputTensorShapesStr,
            const vector<armnn::BackendId>& computeDevice,
            const std::string& dynamicBackendsPath,
            const std::string& path,
            const std::string& inputNames,
            const std::string& inputTensorDataFilePaths,
            const std::string& inputTypes,
            bool quantizeInput,
            const std::string& outputTypes,
            const std::string& outputNames,
            bool enableProfiling,
            bool enableFp16TurboMode,
            const double& thresholdTime,
            const size_t subgraphId,
            const std::shared_ptr<armnn::IRuntime>& runtime = nullptr)
{
    std::string modelFormat = boost::trim_copy(format);
    std::string modelPath = boost::trim_copy(path);
    std::vector<std::string> inputNamesVector = ParseStringList(inputNames, ",");
    std::vector<std::string> inputTensorShapesVector = ParseStringList(inputTensorShapesStr, ";");
    std::vector<std::string> inputTensorDataFilePathsVector = ParseStringList(
        inputTensorDataFilePaths, ",");
    std::vector<std::string> outputNamesVector = ParseStringList(outputNames, ",");
    std::vector<std::string> inputTypesVector = ParseStringList(inputTypes, ",");
    std::vector<std::string> outputTypesVector = ParseStringList(outputTypes, ",");

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
        return EXIT_FAILURE;
    }

    if ((inputTensorShapesVector.size() != 0) && (inputTensorShapesVector.size() != inputNamesVector.size()))
    {
        BOOST_LOG_TRIVIAL(fatal) << "input-name and input-tensor-shape must have the same amount of elements.";
        return EXIT_FAILURE;
    }

    if ((inputTensorDataFilePathsVector.size() != 0) &&
        (inputTensorDataFilePathsVector.size() != inputNamesVector.size()))
    {
        BOOST_LOG_TRIVIAL(fatal) << "input-name and input-tensor-data must have the same amount of elements.";
        return EXIT_FAILURE;
    }

    if (inputTypesVector.size() == 0)
    {
        //Defaults the value of all inputs to "float"
        inputTypesVector.assign(inputNamesVector.size(), "float");
    }
    if (outputTypesVector.size() == 0)
    {
        //Defaults the value of all outputs to "float"
        outputTypesVector.assign(outputNamesVector.size(), "float");
    }
    else if ((inputTypesVector.size() != 0) && (inputTypesVector.size() != inputNamesVector.size()))
    {
        BOOST_LOG_TRIVIAL(fatal) << "input-name and input-type must have the same amount of elements.";
        return EXIT_FAILURE;
    }

    // Parse input tensor shape from the string we got from the command-line.
    std::vector<std::unique_ptr<armnn::TensorShape>> inputTensorShapes;

    if (!inputTensorShapesVector.empty())
    {
        inputTensorShapes.reserve(inputTensorShapesVector.size());

        for(const std::string& shape : inputTensorShapesVector)
        {
            std::stringstream ss(shape);
            std::vector<unsigned int> dims = ParseArray(ss);

            try
            {
                // Coverity fix: An exception of type armnn::InvalidArgumentException is thrown and never caught.
                inputTensorShapes.push_back(std::make_unique<armnn::TensorShape>(dims.size(), dims.data()));
            }
            catch (const armnn::InvalidArgumentException& e)
            {
                BOOST_LOG_TRIVIAL(fatal) << "Cannot create tensor shape: " << e.what();
                return EXIT_FAILURE;
            }
        }
    }

    // Check that threshold time is not less than zero
    if (thresholdTime < 0)
    {
        BOOST_LOG_TRIVIAL(fatal) << "Threshold time supplied as a commoand line argument is less than zero.";
        return EXIT_FAILURE;
    }

    // Forward to implementation based on the parser type
    if (modelFormat.find("armnn") != std::string::npos)
    {
#if defined(ARMNN_SERIALIZER)
    return MainImpl<armnnDeserializer::IDeserializer, float>(
        modelPath.c_str(), isModelBinary, computeDevice,
        dynamicBackendsPath, inputNamesVector, inputTensorShapes,
        inputTensorDataFilePathsVector, inputTypesVector, quantizeInput,
        outputTypesVector, outputNamesVector, enableProfiling,
        enableFp16TurboMode, thresholdTime, subgraphId, runtime);
#else
    BOOST_LOG_TRIVIAL(fatal) << "Not built with serialization support.";
    return EXIT_FAILURE;
#endif
    }
    else if (modelFormat.find("caffe") != std::string::npos)
    {
#if defined(ARMNN_CAFFE_PARSER)
        return MainImpl<armnnCaffeParser::ICaffeParser, float>(modelPath.c_str(), isModelBinary, computeDevice,
                                                               dynamicBackendsPath,
                                                               inputNamesVector, inputTensorShapes,
                                                               inputTensorDataFilePathsVector, inputTypesVector,
                                                               quantizeInput, outputTypesVector, outputNamesVector,
                                                               enableProfiling, enableFp16TurboMode, thresholdTime,
                                                               subgraphId, runtime);
#else
        BOOST_LOG_TRIVIAL(fatal) << "Not built with Caffe parser support.";
        return EXIT_FAILURE;
#endif
    }
    else if (modelFormat.find("onnx") != std::string::npos)
{
#if defined(ARMNN_ONNX_PARSER)
    return MainImpl<armnnOnnxParser::IOnnxParser, float>(modelPath.c_str(), isModelBinary, computeDevice,
                                                         dynamicBackendsPath,
                                                         inputNamesVector, inputTensorShapes,
                                                         inputTensorDataFilePathsVector, inputTypesVector,
                                                         quantizeInput, outputTypesVector, outputNamesVector,
                                                         enableProfiling, enableFp16TurboMode, thresholdTime,
                                                         subgraphId, runtime);
#else
    BOOST_LOG_TRIVIAL(fatal) << "Not built with Onnx parser support.";
    return EXIT_FAILURE;
#endif
    }
    else if (modelFormat.find("tensorflow") != std::string::npos)
    {
#if defined(ARMNN_TF_PARSER)
        return MainImpl<armnnTfParser::ITfParser, float>(modelPath.c_str(), isModelBinary, computeDevice,
                                                         dynamicBackendsPath,
                                                         inputNamesVector, inputTensorShapes,
                                                         inputTensorDataFilePathsVector, inputTypesVector,
                                                         quantizeInput, outputTypesVector, outputNamesVector,
                                                         enableProfiling, enableFp16TurboMode, thresholdTime,
                                                         subgraphId, runtime);
#else
        BOOST_LOG_TRIVIAL(fatal) << "Not built with Tensorflow parser support.";
        return EXIT_FAILURE;
#endif
    }
    else if(modelFormat.find("tflite") != std::string::npos)
    {
#if defined(ARMNN_TF_LITE_PARSER)
        if (! isModelBinary)
        {
            BOOST_LOG_TRIVIAL(fatal) << "Unknown model format: '" << modelFormat << "'. Only 'binary' format supported \
              for tflite files";
            return EXIT_FAILURE;
        }
        return MainImpl<armnnTfLiteParser::ITfLiteParser, float>(modelPath.c_str(), isModelBinary, computeDevice,
                                                                 dynamicBackendsPath,
                                                                 inputNamesVector, inputTensorShapes,
                                                                 inputTensorDataFilePathsVector, inputTypesVector,
                                                                 quantizeInput, outputTypesVector, outputNamesVector,
                                                                 enableProfiling, enableFp16TurboMode, thresholdTime,
                                                                 subgraphId, runtime);
#else
        BOOST_LOG_TRIVIAL(fatal) << "Unknown model format: '" << modelFormat <<
            "'. Please include 'caffe', 'tensorflow', 'tflite' or 'onnx'";
        return EXIT_FAILURE;
#endif
    }
    else
    {
        BOOST_LOG_TRIVIAL(fatal) << "Unknown model format: '" << modelFormat <<
                                 "'. Please include 'caffe', 'tensorflow', 'tflite' or 'onnx'";
        return EXIT_FAILURE;
    }
}

int RunCsvTest(const armnnUtils::CsvRow &csvRow, const std::shared_ptr<armnn::IRuntime>& runtime,
               const bool enableProfiling, const bool enableFp16TurboMode, const double& thresholdTime)
{
    std::string modelFormat;
    std::string modelPath;
    std::string inputNames;
    std::string inputTensorShapes;
    std::string inputTensorDataFilePaths;
    std::string outputNames;
    std::string inputTypes;
    std::string outputTypes;
    std::string dynamicBackendsPath;

    size_t subgraphId = 0;

    const std::string backendsMessage = std::string("The preferred order of devices to run layers on by default. ")
                                      + std::string("Possible choices: ")
                                      + armnn::BackendRegistryInstance().GetBackendIdsAsString();

    po::options_description desc("Options");
    try
    {
        desc.add_options()
        ("model-format,f", po::value(&modelFormat),
         "armnn-binary, caffe-binary, caffe-text, tflite-binary, onnx-binary, onnx-text, tensorflow-binary or "
         "tensorflow-text.")
        ("model-path,m", po::value(&modelPath), "Path to model file, e.g. .armnn, .caffemodel, .prototxt, "
         ".tflite, .onnx")
        ("compute,c", po::value<std::vector<armnn::BackendId>>()->multitoken(),
         backendsMessage.c_str())
        ("dynamic-backends-path,b", po::value(&dynamicBackendsPath),
         "Path where to load any available dynamic backend from. "
         "If left empty (the default), dynamic backends will not be used.")
        ("input-name,i", po::value(&inputNames), "Identifier of the input tensors in the network separated by comma.")
        ("subgraph-number,n", po::value<size_t>(&subgraphId)->default_value(0), "Id of the subgraph to be "
         "executed. Defaults to 0.")
        ("input-tensor-shape,s", po::value(&inputTensorShapes),
         "The shape of the input tensors in the network as a flat array of integers separated by comma. "
         "Several shapes can be passed separating them by semicolon. "
         "This parameter is optional, depending on the network.")
        ("input-tensor-data,d", po::value(&inputTensorDataFilePaths),
         "Path to files containing the input data as a flat array separated by whitespace. "
         "Several paths can be passed separating them by comma.")
        ("input-type,y",po::value(&inputTypes), "The type of the input tensors in the network separated by comma. "
         "If unset, defaults to \"float\" for all defined inputs. "
         "Accepted values (float, int or qasymm8).")
        ("quantize-input,q",po::bool_switch()->default_value(false),
         "If this option is enabled, all float inputs will be quantized to qasymm8. "
         "If unset, default to not quantized. "
         "Accepted values (true or false)")
        ("output-type,z",po::value(&outputTypes), "The type of the output tensors in the network separated by comma. "
         "If unset, defaults to \"float\" for all defined outputs. "
         "Accepted values (float, int or qasymm8).")
        ("output-name,o", po::value(&outputNames),
         "Identifier of the output tensors in the network separated by comma.");
    }
    catch (const std::exception& e)
    {
        // Coverity points out that default_value(...) can throw a bad_lexical_cast,
        // and that desc.add_options() can throw boost::io::too_few_args.
        // They really won't in any of these cases.
        BOOST_ASSERT_MSG(false, "Caught unexpected exception");
        BOOST_LOG_TRIVIAL(fatal) << "Fatal internal error: " << e.what();
        return EXIT_FAILURE;
    }

    std::vector<const char*> clOptions;
    clOptions.reserve(csvRow.values.size());
    for (const std::string& value : csvRow.values)
    {
        clOptions.push_back(value.c_str());
    }

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(static_cast<int>(clOptions.size()), clOptions.data(), desc), vm);

        po::notify(vm);

        CheckOptionDependencies(vm);
    }
    catch (const po::error& e)
    {
        std::cerr << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
        return EXIT_FAILURE;
    }

    // Get the value of the switch arguments.
    bool quantizeInput = vm["quantize-input"].as<bool>();

    // Get the preferred order of compute devices.
    std::vector<armnn::BackendId> computeDevices = vm["compute"].as<std::vector<armnn::BackendId>>();

    // Remove duplicates from the list of compute devices.
    RemoveDuplicateDevices(computeDevices);

    // Check that the specified compute devices are valid.
    std::string invalidBackends;
    if (!CheckRequestedBackendsAreValid(computeDevices, armnn::Optional<std::string&>(invalidBackends)))
    {
        BOOST_LOG_TRIVIAL(fatal) << "The list of preferred devices contains invalid backend IDs: "
                                 << invalidBackends;
        return EXIT_FAILURE;
    }

    return RunTest(modelFormat, inputTensorShapes, computeDevices, dynamicBackendsPath, modelPath, inputNames,
                   inputTensorDataFilePaths, inputTypes, quantizeInput, outputTypes, outputNames,
                   enableProfiling, enableFp16TurboMode, thresholdTime, subgraphId);
}
