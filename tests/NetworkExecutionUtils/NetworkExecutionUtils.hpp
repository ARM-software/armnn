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

#include <Profiling.hpp>
#include <ResolveType.hpp>

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
            ARMNN_LOG(error) << "An error occurred when splitting tokens: " << e.what();
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
                    ARMNN_LOG(error) << "'" << token << "' is not a valid number. It has been ignored.";
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
auto ParseDataArray<armnn::DataType::QAsymmU8>(std::istream& stream)
{
    return ParseArrayImpl<uint8_t>(stream,
                                   [](const std::string& s) { return boost::numeric_cast<uint8_t>(std::stoi(s)); });
}

template<>
auto ParseDataArray<armnn::DataType::QAsymmU8>(std::istream& stream,
                                                      const float& quantizationScale,
                                                      const int32_t& quantizationOffset)
{
    return ParseArrayImpl<uint8_t>(stream,
                                   [&quantizationScale, &quantizationOffset](const std::string & s)
                                   {
                                       return boost::numeric_cast<uint8_t>(
                                           armnn::Quantize<uint8_t>(std::stof(s),
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
    TensorPrinter(const std::string& binding,
                  const armnn::TensorInfo& info,
                  const std::string& outputTensorFile,
                  bool dequantizeOutput)
        : m_OutputBinding(binding)
        , m_Scale(info.GetQuantizationScale())
        , m_Offset(info.GetQuantizationOffset())
        , m_OutputTensorFile(outputTensorFile)
        , m_DequantizeOutput(dequantizeOutput)
    {}

    void operator()(const std::vector<float>& values)
    {
        ForEachValue(values, [](float value)
            {
                printf("%f ", value);
            });
        WriteToFile(values);
    }

    void operator()(const std::vector<uint8_t>& values)
    {
        if(m_DequantizeOutput)
        {
            auto& scale = m_Scale;
            auto& offset = m_Offset;
            std::vector<float> dequantizedValues;
            ForEachValue(values, [&scale, &offset, &dequantizedValues](uint8_t value)
            {
                auto dequantizedValue = armnn::Dequantize(value, scale, offset);
                printf("%f ", dequantizedValue);
                dequantizedValues.push_back(dequantizedValue);
            });
            WriteToFile(dequantizedValues);
        }
        else
        {
            const std::vector<int> intValues(values.begin(), values.end());
            operator()(intValues);
        }
    }

    void operator()(const std::vector<int>& values)
    {
        ForEachValue(values, [](int value)
            {
                printf("%d ", value);
            });
        WriteToFile(values);
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

    template<typename T>
    void WriteToFile(const std::vector<T>& values)
    {
        if (!m_OutputTensorFile.empty())
        {
            std::ofstream outputTensorFile;
            outputTensorFile.open(m_OutputTensorFile, std::ofstream::out | std::ofstream::trunc);
            if (outputTensorFile.is_open())
            {
                outputTensorFile << m_OutputBinding << ": ";
                std::copy(values.begin(), values.end(), std::ostream_iterator<T>(outputTensorFile, " "));
            }
            else
            {
                ARMNN_LOG(info) << "Output Tensor File: " << m_OutputTensorFile << " could not be opened!";
            }
            outputTensorFile.close();
        }
    }

    std::string m_OutputBinding;
    float m_Scale=0.0f;
    int m_Offset=0;
    std::string m_OutputTensorFile;
    bool m_DequantizeOutput = false;
};



template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
std::vector<T> GenerateDummyTensorData(unsigned int numElements)
{
    return std::vector<T>(numElements, static_cast<T>(0));
}

using TContainer         = boost::variant<std::vector<float>, std::vector<int>, std::vector<unsigned char>>;
using QuantizationParams = std::pair<float, int32_t>;

void PopulateTensorWithData(TContainer& tensorData,
                            unsigned int numElements,
                            const std::string& dataTypeStr,
                            const armnn::Optional<QuantizationParams>& qParams,
                            const armnn::Optional<std::string>& dataFile)
{
    const bool readFromFile = dataFile.has_value() && !dataFile.value().empty();
    const bool quantizeData = qParams.has_value();

    std::ifstream inputTensorFile;
    if (readFromFile)
    {
        inputTensorFile = std::ifstream(dataFile.value());
    }

    if (dataTypeStr.compare("float") == 0)
    {
        if (quantizeData)
        {
            const float qScale  = qParams.value().first;
            const int   qOffset = qParams.value().second;

            tensorData = readFromFile ?
                ParseDataArray<armnn::DataType::QAsymmU8>(inputTensorFile, qScale, qOffset) :
                GenerateDummyTensorData<armnn::DataType::QAsymmU8>(numElements);
        }
        else
        {
            tensorData = readFromFile ?
                ParseDataArray<armnn::DataType::Float32>(inputTensorFile) :
                GenerateDummyTensorData<armnn::DataType::Float32>(numElements);
        }
    }
    else if (dataTypeStr.compare("int") == 0)
    {
        tensorData = readFromFile ?
            ParseDataArray<armnn::DataType::Signed32>(inputTensorFile) :
            GenerateDummyTensorData<armnn::DataType::Signed32>(numElements);
    }
    else if (dataTypeStr.compare("qasymm8") == 0)
    {
         tensorData = readFromFile ?
            ParseDataArray<armnn::DataType::QAsymmU8>(inputTensorFile) :
            GenerateDummyTensorData<armnn::DataType::QAsymmU8>(numElements);
    }
    else
    {
        std::string errorMessage = "Unsupported tensor data type " + dataTypeStr;
        ARMNN_LOG(fatal) << errorMessage;

        inputTensorFile.close();
        throw armnn::Exception(errorMessage);
    }

    inputTensorFile.close();
}

} // anonymous namespace

bool generateTensorData = true;

struct ExecuteNetworkParams
{
    using TensorShapePtr = std::unique_ptr<armnn::TensorShape>;

    const char*                   m_ModelPath;
    bool                          m_IsModelBinary;
    std::vector<armnn::BackendId> m_ComputeDevices;
    std::string                   m_DynamicBackendsPath;
    std::vector<string>           m_InputNames;
    std::vector<TensorShapePtr>   m_InputTensorShapes;
    std::vector<string>           m_InputTensorDataFilePaths;
    std::vector<string>           m_InputTypes;
    bool                          m_QuantizeInput;
    std::vector<string>           m_OutputTypes;
    std::vector<string>           m_OutputNames;
    std::vector<string>           m_OutputTensorFiles;
    bool                          m_DequantizeOutput;
    bool                          m_EnableProfiling;
    bool                          m_EnableFp16TurboMode;
    double                        m_ThresholdTime;
    bool                          m_PrintIntermediate;
    size_t                        m_SubgraphId;
    bool                          m_EnableLayerDetails = false;
    bool                          m_GenerateTensorData;
    bool                          m_ParseUnsupported = false;
};

template<typename TParser, typename TDataType>
int MainImpl(const ExecuteNetworkParams& params,
             const std::shared_ptr<armnn::IRuntime>& runtime = nullptr)
{
    using TContainer = boost::variant<std::vector<float>, std::vector<int>, std::vector<unsigned char>>;

    std::vector<TContainer> inputDataContainers;

    try
    {
        // Creates an InferenceModel, which will parse the model and load it into an IRuntime.
        typename InferenceModel<TParser, TDataType>::Params inferenceModelParams;
        inferenceModelParams.m_ModelPath                      = params.m_ModelPath;
        inferenceModelParams.m_IsModelBinary                  = params.m_IsModelBinary;
        inferenceModelParams.m_ComputeDevices                 = params.m_ComputeDevices;
        inferenceModelParams.m_DynamicBackendsPath            = params.m_DynamicBackendsPath;
        inferenceModelParams.m_PrintIntermediateLayers        = params.m_PrintIntermediate;
        inferenceModelParams.m_VisualizePostOptimizationModel = params.m_EnableLayerDetails;
        inferenceModelParams.m_ParseUnsupported               = params.m_ParseUnsupported;

        for(const std::string& inputName: params.m_InputNames)
        {
            inferenceModelParams.m_InputBindings.push_back(inputName);
        }

        for(unsigned int i = 0; i < params.m_InputTensorShapes.size(); ++i)
        {
            inferenceModelParams.m_InputShapes.push_back(*params.m_InputTensorShapes[i]);
        }

        for(const std::string& outputName: params.m_OutputNames)
        {
            inferenceModelParams.m_OutputBindings.push_back(outputName);
        }

        inferenceModelParams.m_SubgraphId          = params.m_SubgraphId;
        inferenceModelParams.m_EnableFp16TurboMode = params.m_EnableFp16TurboMode;

        InferenceModel<TParser, TDataType> model(inferenceModelParams,
                                                 params.m_EnableProfiling,
                                                 params.m_DynamicBackendsPath,
                                                 runtime);

        const size_t numInputs = inferenceModelParams.m_InputBindings.size();
        for(unsigned int i = 0; i < numInputs; ++i)
        {
            armnn::Optional<QuantizationParams> qParams = params.m_QuantizeInput ?
                armnn::MakeOptional<QuantizationParams>(model.GetInputQuantizationParams()) :
                armnn::EmptyOptional();

            armnn::Optional<std::string> dataFile = params.m_GenerateTensorData ?
                armnn::EmptyOptional() :
                armnn::MakeOptional<std::string>(params.m_InputTensorDataFilePaths[i]);

            unsigned int numElements = model.GetInputSize(i);
            if (params.m_InputTensorShapes.size() > i && params.m_InputTensorShapes[i])
            {
                // If the user has provided a tensor shape for the current input,
                // override numElements
                numElements = params.m_InputTensorShapes[i]->GetNumElements();
            }

            TContainer tensorData;
            PopulateTensorWithData(tensorData,
                                   numElements,
                                   params.m_InputTypes[i],
                                   qParams,
                                   dataFile);

            inputDataContainers.push_back(tensorData);
        }

        const size_t numOutputs = inferenceModelParams.m_OutputBindings.size();
        std::vector<TContainer> outputDataContainers;

        for (unsigned int i = 0; i < numOutputs; ++i)
        {
            if (params.m_OutputTypes[i].compare("float") == 0)
            {
                outputDataContainers.push_back(std::vector<float>(model.GetOutputSize(i)));
            }
            else if (params.m_OutputTypes[i].compare("int") == 0)
            {
                outputDataContainers.push_back(std::vector<int>(model.GetOutputSize(i)));
            }
            else if (params.m_OutputTypes[i].compare("qasymm8") == 0)
            {
                outputDataContainers.push_back(std::vector<uint8_t>(model.GetOutputSize(i)));
            }
            else
            {
                ARMNN_LOG(fatal) << "Unsupported tensor data type \"" << params.m_OutputTypes[i] << "\". ";
                return EXIT_FAILURE;
            }
        }

        // model.Run returns the inference time elapsed in EnqueueWorkload (in milliseconds)
        auto inference_duration = model.Run(inputDataContainers, outputDataContainers);

        if (params.m_GenerateTensorData)
        {
            ARMNN_LOG(warning) << "The input data was generated, note that the output will not be useful";
        }

        // Print output tensors
        const auto& infosOut = model.GetOutputBindingInfos();
        for (size_t i = 0; i < numOutputs; i++)
        {
            const armnn::TensorInfo& infoOut = infosOut[i].second;
            auto outputTensorFile = params.m_OutputTensorFiles.empty() ? "" : params.m_OutputTensorFiles[i];

            TensorPrinter printer(inferenceModelParams.m_OutputBindings[i],
                                  infoOut,
                                  outputTensorFile,
                                  params.m_DequantizeOutput);
            boost::apply_visitor(printer, outputDataContainers[i]);
        }

        ARMNN_LOG(info) << "\nInference time: " << std::setprecision(2)
                                << std::fixed << inference_duration.count() << " ms";

        // If thresholdTime == 0.0 (default), then it hasn't been supplied at command line
        if (params.m_ThresholdTime != 0.0)
        {
            ARMNN_LOG(info) << "Threshold time: " << std::setprecision(2)
                                    << std::fixed << params.m_ThresholdTime << " ms";
            auto thresholdMinusInference = params.m_ThresholdTime - inference_duration.count();
            ARMNN_LOG(info) << "Threshold time - Inference time: " << std::setprecision(2)
                                    << std::fixed << thresholdMinusInference << " ms" << "\n";

            if (thresholdMinusInference < 0)
            {
                std::string errorMessage = "Elapsed inference time is greater than provided threshold time.";
                ARMNN_LOG(fatal) << errorMessage;
            }
        }
    }
    catch (armnn::Exception const& e)
    {
        ARMNN_LOG(fatal) << "Armnn Error: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// This will run a test
int RunTest(const std::string& format,
            const std::string& inputTensorShapesStr,
            const vector<armnn::BackendId>& computeDevices,
            const std::string& dynamicBackendsPath,
            const std::string& path,
            const std::string& inputNames,
            const std::string& inputTensorDataFilePaths,
            const std::string& inputTypes,
            bool quantizeInput,
            const std::string& outputTypes,
            const std::string& outputNames,
            const std::string& outputTensorFiles,
            bool dequantizeOuput,
            bool enableProfiling,
            bool enableFp16TurboMode,
            const double& thresholdTime,
            bool printIntermediate,
            const size_t subgraphId,
            bool enableLayerDetails = false,
            bool parseUnsupported = false,
            const std::shared_ptr<armnn::IRuntime>& runtime = nullptr)
{
    std::string modelFormat = boost::trim_copy(format);
    std::string modelPath = boost::trim_copy(path);
    std::vector<std::string> inputNamesVector = ParseStringList(inputNames, ",");
    std::vector<std::string> inputTensorShapesVector = ParseStringList(inputTensorShapesStr, ":");
    std::vector<std::string> inputTensorDataFilePathsVector = ParseStringList(
        inputTensorDataFilePaths, ",");
    std::vector<std::string> outputNamesVector = ParseStringList(outputNames, ",");
    std::vector<std::string> inputTypesVector = ParseStringList(inputTypes, ",");
    std::vector<std::string> outputTypesVector = ParseStringList(outputTypes, ",");
    std::vector<std::string> outputTensorFilesVector = ParseStringList(outputTensorFiles, ",");

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
        ARMNN_LOG(fatal) << "Unknown model format: '" << modelFormat << "'. Please include 'binary' or 'text'";
        return EXIT_FAILURE;
    }

    if ((inputTensorShapesVector.size() != 0) && (inputTensorShapesVector.size() != inputNamesVector.size()))
    {
        ARMNN_LOG(fatal) << "input-name and input-tensor-shape must have the same amount of elements.";
        return EXIT_FAILURE;
    }

    if ((inputTensorDataFilePathsVector.size() != 0) &&
        (inputTensorDataFilePathsVector.size() != inputNamesVector.size()))
    {
        ARMNN_LOG(fatal) << "input-name and input-tensor-data must have the same amount of elements.";
        return EXIT_FAILURE;
    }

    if ((outputTensorFilesVector.size() != 0) &&
        (outputTensorFilesVector.size() != outputNamesVector.size()))
    {
        ARMNN_LOG(fatal) << "output-name and write-outputs-to-file must have the same amount of elements.";
        return EXIT_FAILURE;
    }

    if (inputTypesVector.size() == 0)
    {
        //Defaults the value of all inputs to "float"
        inputTypesVector.assign(inputNamesVector.size(), "float");
    }
    else if ((inputTypesVector.size() != 0) && (inputTypesVector.size() != inputNamesVector.size()))
    {
        ARMNN_LOG(fatal) << "input-name and input-type must have the same amount of elements.";
        return EXIT_FAILURE;
    }

    if (outputTypesVector.size() == 0)
    {
        //Defaults the value of all outputs to "float"
        outputTypesVector.assign(outputNamesVector.size(), "float");
    }
    else if ((outputTypesVector.size() != 0) && (outputTypesVector.size() != outputNamesVector.size()))
    {
        ARMNN_LOG(fatal) << "output-name and output-type must have the same amount of elements.";
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
                ARMNN_LOG(fatal) << "Cannot create tensor shape: " << e.what();
                return EXIT_FAILURE;
            }
        }
    }

    // Check that threshold time is not less than zero
    if (thresholdTime < 0)
    {
        ARMNN_LOG(fatal) << "Threshold time supplied as a command line argument is less than zero.";
        return EXIT_FAILURE;
    }

    ExecuteNetworkParams params;
    params.m_ModelPath                = modelPath.c_str();
    params.m_IsModelBinary            = isModelBinary;
    params.m_ComputeDevices           = computeDevices;
    params.m_DynamicBackendsPath      = dynamicBackendsPath;
    params.m_InputNames               = inputNamesVector;
    params.m_InputTensorShapes        = std::move(inputTensorShapes);
    params.m_InputTensorDataFilePaths = inputTensorDataFilePathsVector;
    params.m_InputTypes               = inputTypesVector;
    params.m_QuantizeInput            = quantizeInput;
    params.m_OutputTypes              = outputTypesVector;
    params.m_OutputNames              = outputNamesVector;
    params.m_OutputTensorFiles        = outputTensorFilesVector;
    params.m_DequantizeOutput         = dequantizeOuput;
    params.m_EnableProfiling          = enableProfiling;
    params.m_EnableFp16TurboMode      = enableFp16TurboMode;
    params.m_ThresholdTime            = thresholdTime;
    params.m_PrintIntermediate        = printIntermediate;
    params.m_SubgraphId               = subgraphId;
    params.m_EnableLayerDetails       = enableLayerDetails;
    params.m_GenerateTensorData       = inputTensorDataFilePathsVector.empty();
    params.m_ParseUnsupported         = parseUnsupported;

    // Warn if ExecuteNetwork will generate dummy input data
    if (params.m_GenerateTensorData)
    {
        ARMNN_LOG(warning) << "No input files provided, input tensors will be filled with 0s.";
    }

    // Forward to implementation based on the parser type
    if (modelFormat.find("armnn") != std::string::npos)
    {
#if defined(ARMNN_SERIALIZER)
    return MainImpl<armnnDeserializer::IDeserializer, float>(params, runtime);
#else
        ARMNN_LOG(fatal) << "Not built with serialization support.";
    return EXIT_FAILURE;
#endif
    }
    else if (modelFormat.find("caffe") != std::string::npos)
    {
#if defined(ARMNN_CAFFE_PARSER)
        return MainImpl<armnnCaffeParser::ICaffeParser, float>(params, runtime);
#else
        ARMNN_LOG(fatal) << "Not built with Caffe parser support.";
        return EXIT_FAILURE;
#endif
    }
    else if (modelFormat.find("onnx") != std::string::npos)
{
#if defined(ARMNN_ONNX_PARSER)
    return MainImpl<armnnOnnxParser::IOnnxParser, float>(params, runtime);
#else
        ARMNN_LOG(fatal) << "Not built with Onnx parser support.";
    return EXIT_FAILURE;
#endif
    }
    else if (modelFormat.find("tensorflow") != std::string::npos)
    {
#if defined(ARMNN_TF_PARSER)
        return MainImpl<armnnTfParser::ITfParser, float>(params, runtime);
#else
        ARMNN_LOG(fatal) << "Not built with Tensorflow parser support.";
        return EXIT_FAILURE;
#endif
    }
    else if(modelFormat.find("tflite") != std::string::npos)
    {
#if defined(ARMNN_TF_LITE_PARSER)
        if (! isModelBinary)
        {
            ARMNN_LOG(fatal) << "Unknown model format: '" << modelFormat << "'. Only 'binary' format supported \
              for tflite files";
            return EXIT_FAILURE;
        }
        return MainImpl<armnnTfLiteParser::ITfLiteParser, float>(params, runtime);
#else
        ARMNN_LOG(fatal) << "Unknown model format: '" << modelFormat <<
            "'. Please include 'caffe', 'tensorflow', 'tflite' or 'onnx'";
        return EXIT_FAILURE;
#endif
    }
    else
    {
        ARMNN_LOG(fatal) << "Unknown model format: '" << modelFormat <<
                                 "'. Please include 'caffe', 'tensorflow', 'tflite' or 'onnx'";
        return EXIT_FAILURE;
    }
}

int RunCsvTest(const armnnUtils::CsvRow &csvRow, const std::shared_ptr<armnn::IRuntime>& runtime,
               const bool enableProfiling, const bool enableFp16TurboMode, const double& thresholdTime,
               const bool printIntermediate, bool enableLayerDetails = false, bool parseUnuspported = false)
{
    boost::ignore_unused(runtime);
    std::string modelFormat;
    std::string modelPath;
    std::string inputNames;
    std::string inputTensorShapes;
    std::string inputTensorDataFilePaths;
    std::string outputNames;
    std::string inputTypes;
    std::string outputTypes;
    std::string dynamicBackendsPath;
    std::string outputTensorFiles;

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
        ("input-tensor-data,d", po::value(&inputTensorDataFilePaths)->default_value(""),
         "Path to files containing the input data as a flat array separated by whitespace. "
         "Several paths can be passed separating them by comma. If not specified, the network will be run with dummy "
         "data (useful for profiling).")
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
         "Identifier of the output tensors in the network separated by comma.")
        ("dequantize-output,l",po::bool_switch()->default_value(false),
         "If this option is enabled, all quantized outputs will be dequantized to float. "
         "If unset, default to not get dequantized. "
         "Accepted values (true or false)")
        ("write-outputs-to-file,w", po::value(&outputTensorFiles),
         "Comma-separated list of output file paths keyed with the binding-id of the output slot. "
         "If left empty (the default), the output tensors will not be written to a file.");
    }
    catch (const std::exception& e)
    {
        // Coverity points out that default_value(...) can throw a bad_lexical_cast,
        // and that desc.add_options() can throw boost::io::too_few_args.
        // They really won't in any of these cases.
        BOOST_ASSERT_MSG(false, "Caught unexpected exception");
        ARMNN_LOG(fatal) << "Fatal internal error: " << e.what();
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
    bool dequantizeOutput = vm["dequantize-output"].as<bool>();

    // Get the preferred order of compute devices.
    std::vector<armnn::BackendId> computeDevices = vm["compute"].as<std::vector<armnn::BackendId>>();

    // Remove duplicates from the list of compute devices.
    RemoveDuplicateDevices(computeDevices);

    // Check that the specified compute devices are valid.
    std::string invalidBackends;
    if (!CheckRequestedBackendsAreValid(computeDevices, armnn::Optional<std::string&>(invalidBackends)))
    {
        ARMNN_LOG(fatal) << "The list of preferred devices contains invalid backend IDs: "
                                 << invalidBackends;
        return EXIT_FAILURE;
    }

    return RunTest(modelFormat, inputTensorShapes, computeDevices, dynamicBackendsPath, modelPath, inputNames,
                   inputTensorDataFilePaths, inputTypes, quantizeInput, outputTypes, outputNames, outputTensorFiles,
                   dequantizeOutput, enableProfiling, enableFp16TurboMode, thresholdTime, printIntermediate, subgraphId,
                   enableLayerDetails, parseUnuspported);
}
