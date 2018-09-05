//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "armnn/ArmNN.hpp"

#include <armnn/TypesUtils.hpp>

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
std::vector<T> ParseArrayImpl(std::istream& stream, TParseElementFunc parseElementFunc)
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

void RemoveDuplicateDevices(std::vector<armnn::Compute>& computeDevices)
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

bool CheckDevicesAreValid(const std::vector<armnn::Compute>& computeDevices)
{
    return (!computeDevices.empty()
            && std::none_of(computeDevices.begin(), computeDevices.end(),
                            [](armnn::Compute c){ return c == armnn::Compute::Undefined; }));
}

} // namespace

template<typename TParser, typename TDataType>
int MainImpl(const char* modelPath,
             bool isModelBinary,
             const std::vector<armnn::Compute>& computeDevice,
             const char* inputName,
             const armnn::TensorShape* inputTensorShape,
             const char* inputTensorDataFilePath,
             const char* outputName,
             bool enableProfiling,
             const size_t subgraphId,
             const std::shared_ptr<armnn::IRuntime>& runtime = nullptr)
{
    // Loads input tensor.
    std::vector<TDataType> input;
    {
        std::ifstream inputTensorFile(inputTensorDataFilePath);
        if (!inputTensorFile.good())
        {
            BOOST_LOG_TRIVIAL(fatal) << "Failed to load input tensor data file from " << inputTensorDataFilePath;
            return EXIT_FAILURE;
        }
        input = ParseArray<TDataType>(inputTensorFile);
    }

    try
    {
        // Creates an InferenceModel, which will parse the model and load it into an IRuntime.
        typename InferenceModel<TParser, TDataType>::Params params;
        params.m_ModelPath = modelPath;
        params.m_IsModelBinary = isModelBinary;
        params.m_ComputeDevice = computeDevice;
        params.m_InputBinding = inputName;
        params.m_InputTensorShape = inputTensorShape;
        params.m_OutputBinding = outputName;
        params.m_EnableProfiling = enableProfiling;
        params.m_SubgraphId = subgraphId;
        InferenceModel<TParser, TDataType> model(params, runtime);

        // Executes the model.
        std::vector<TDataType> output(model.GetOutputSize());
        model.Run(input, output);

        // Prints the output tensor.
        PrintArray(output);
    }
    catch (armnn::Exception const& e)
    {
        BOOST_LOG_TRIVIAL(fatal) << "Armnn Error: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// This will run a test
int RunTest(const std::string& modelFormat,
            const std::string& inputTensorShapeStr,
            const vector<armnn::Compute>& computeDevice,
            const std::string& modelPath,
            const std::string& inputName,
            const std::string& inputTensorDataFilePath,
            const std::string& outputName,
            bool enableProfiling,
            const size_t subgraphId,
            const std::shared_ptr<armnn::IRuntime>& runtime = nullptr)
{
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
            return EXIT_FAILURE;
        }
    }

    // Forward to implementation based on the parser type
    if (modelFormat.find("caffe") != std::string::npos)
    {
#if defined(ARMNN_CAFFE_PARSER)
        return MainImpl<armnnCaffeParser::ICaffeParser, float>(modelPath.c_str(), isModelBinary, computeDevice,
                                                               inputName.c_str(), inputTensorShape.get(),
                                                               inputTensorDataFilePath.c_str(), outputName.c_str(),
                                                               enableProfiling, subgraphId, runtime);
#else
        BOOST_LOG_TRIVIAL(fatal) << "Not built with Caffe parser support.";
        return EXIT_FAILURE;
#endif
    }
    else if (modelFormat.find("onnx") != std::string::npos)
{
#if defined(ARMNN_ONNX_PARSER)
    return MainImpl<armnnOnnxParser::IOnnxParser, float>(modelPath.c_str(), isModelBinary, computeDevice,
                                                         inputName.c_str(), inputTensorShape.get(),
                                                         inputTensorDataFilePath.c_str(), outputName.c_str(),
                                                         enableProfiling, subgraphId, runtime);
#else
    BOOST_LOG_TRIVIAL(fatal) << "Not built with Onnx parser support.";
    return EXIT_FAILURE;
#endif
    }
    else if (modelFormat.find("tensorflow") != std::string::npos)
    {
#if defined(ARMNN_TF_PARSER)
        return MainImpl<armnnTfParser::ITfParser, float>(modelPath.c_str(), isModelBinary, computeDevice,
                                                         inputName.c_str(), inputTensorShape.get(),
                                                         inputTensorDataFilePath.c_str(), outputName.c_str(),
                                                         enableProfiling, subgraphId, runtime);
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
                                                                 inputName.c_str(), inputTensorShape.get(),
                                                                 inputTensorDataFilePath.c_str(), outputName.c_str(),
                                                                 enableProfiling, subgraphId, runtime);
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

int RunCsvTest(const armnnUtils::CsvRow &csvRow,
               const std::shared_ptr<armnn::IRuntime>& runtime)
{
    std::string modelFormat;
    std::string modelPath;
    std::string inputName;
    std::string inputTensorShapeStr;
    std::string inputTensorDataFilePath;
    std::string outputName;

    size_t subgraphId = 0;

    po::options_description desc("Options");
    try
    {
        desc.add_options()
        ("model-format,f", po::value(&modelFormat),
         "caffe-binary, caffe-text, tflite-binary, onnx-binary, onnx-text, tensorflow-binary or tensorflow-text.")
        ("model-path,m", po::value(&modelPath), "Path to model file, e.g. .caffemodel, .prototxt, .tflite,"
         " .onnx")
        ("compute,c", po::value<std::vector<armnn::Compute>>()->multitoken(),
         "The preferred order of devices to run layers on by default. Possible choices: CpuAcc, CpuRef, GpuAcc")
        ("input-name,i", po::value(&inputName), "Identifier of the input tensor in the network.")
        ("subgraph-number,n", po::value<size_t>(&subgraphId)->default_value(0), "Id of the subgraph to be "
         "executed. Defaults to 0")
        ("input-tensor-shape,s", po::value(&inputTensorShapeStr),
         "The shape of the input tensor in the network as a flat array of integers separated by whitespace. "
         "This parameter is optional, depending on the network.")
        ("input-tensor-data,d", po::value(&inputTensorDataFilePath),
         "Path to a file containing the input data as a flat array separated by whitespace.")
        ("output-name,o", po::value(&outputName), "Identifier of the output tensor in the network.")
        ("event-based-profiling,e", po::bool_switch()->default_value(false),
         "Enables built in profiler. If unset, defaults to off.");
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

    // Remove leading and trailing whitespaces from the parsed arguments.
    boost::trim(modelFormat);
    boost::trim(modelPath);
    boost::trim(inputName);
    boost::trim(inputTensorShapeStr);
    boost::trim(inputTensorDataFilePath);
    boost::trim(outputName);

    // Get the value of the switch arguments.
    bool enableProfiling = vm["event-based-profiling"].as<bool>();

    // Get the preferred order of compute devices.
    std::vector<armnn::Compute> computeDevices = vm["compute"].as<std::vector<armnn::Compute>>();

    // Remove duplicates from the list of compute devices.
    RemoveDuplicateDevices(computeDevices);

    // Check that the specified compute devices are valid.
    if (!CheckDevicesAreValid(computeDevices))
    {
        BOOST_LOG_TRIVIAL(fatal) << "The list of preferred devices contains an invalid compute";
        return EXIT_FAILURE;
    }

    return RunTest(modelFormat, inputTensorShapeStr, computeDevices,
                   modelPath, inputName, inputTensorDataFilePath, outputName, enableProfiling, subgraphId, runtime);
}

int main(int argc, const char* argv[])
{
    // Configures logging for both the ARMNN library and this test program.
#ifdef NDEBUG
    armnn::LogSeverity level = armnn::LogSeverity::Info;
#else
    armnn::LogSeverity level = armnn::LogSeverity::Debug;
#endif
    armnn::ConfigureLogging(true, true, level);
    armnnUtils::ConfigureLogging(boost::log::core::get().get(), true, true, level);

    std::string testCasesFile;

    std::string modelFormat;
    std::string modelPath;
    std::string inputName;
    std::string inputTensorShapeStr;
    std::string inputTensorDataFilePath;
    std::string outputName;

    size_t subgraphId = 0;

    po::options_description desc("Options");
    try
    {
        desc.add_options()
            ("help", "Display usage information")
            ("test-cases,t", po::value(&testCasesFile), "Path to a CSV file containing test cases to run. "
             "If set, further parameters -- with the exception of compute device and concurrency -- will be ignored, "
             "as they are expected to be defined in the file for each test in particular.")
            ("concurrent,n", po::bool_switch()->default_value(false),
             "Whether or not the test cases should be executed in parallel")
            ("model-format,f", po::value(&modelFormat),
             "caffe-binary, caffe-text, onnx-binary, onnx-text, tflite-binary, tensorflow-binary or tensorflow-text.")
            ("model-path,m", po::value(&modelPath), "Path to model file, e.g. .caffemodel, .prototxt,"
             " .tflite, .onnx")
            ("compute,c", po::value<std::vector<armnn::Compute>>()->multitoken(),
             "The preferred order of devices to run layers on by default. Possible choices: CpuAcc, CpuRef, GpuAcc")
            ("input-name,i", po::value(&inputName), "Identifier of the input tensor in the network.")
            ("subgraph-number,x", po::value<size_t>(&subgraphId)->default_value(0), "Id of the subgraph to be executed."
              "Defaults to 0")
            ("input-tensor-shape,s", po::value(&inputTensorShapeStr),
             "The shape of the input tensor in the network as a flat array of integers separated by whitespace. "
             "This parameter is optional, depending on the network.")
            ("input-tensor-data,d", po::value(&inputTensorDataFilePath),
             "Path to a file containing the input data as a flat array separated by whitespace.")
            ("output-name,o", po::value(&outputName), "Identifier of the output tensor in the network.")
            ("event-based-profiling,e", po::bool_switch()->default_value(false),
             "Enables built in profiler. If unset, defaults to off.");
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

    // Parses the command-line.
    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (CheckOption(vm, "help") || argc <= 1)
        {
            std::cout << "Executes a neural network model using the provided input tensor. " << std::endl;
            std::cout << "Prints the resulting output tensor." << std::endl;
            std::cout << std::endl;
            std::cout << desc << std::endl;
            return EXIT_SUCCESS;
        }

        po::notify(vm);
    }
    catch (const po::error& e)
    {
        std::cerr << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
        return EXIT_FAILURE;
    }

    // Get the value of the switch arguments.
    bool concurrent = vm["concurrent"].as<bool>();
    bool enableProfiling = vm["event-based-profiling"].as<bool>();

    // Check whether we have to load test cases from a file.
    if (CheckOption(vm, "test-cases"))
    {
        // Check that the file exists.
        if (!boost::filesystem::exists(testCasesFile))
        {
            BOOST_LOG_TRIVIAL(fatal) << "Given file \"" << testCasesFile << "\" does not exist";
            return EXIT_FAILURE;
        }

        // Parse CSV file and extract test cases
        armnnUtils::CsvReader reader;
        std::vector<armnnUtils::CsvRow> testCases = reader.ParseFile(testCasesFile);

        // Check that there is at least one test case to run
        if (testCases.empty())
        {
            BOOST_LOG_TRIVIAL(fatal) << "Given file \"" << testCasesFile << "\" has no test cases";
            return EXIT_FAILURE;
        }

        // Create runtime
        armnn::IRuntime::CreationOptions options;
        std::shared_ptr<armnn::IRuntime> runtime(armnn::IRuntime::Create(options));

        const std::string executableName("ExecuteNetwork");

        // Check whether we need to run the test cases concurrently
        if (concurrent)
        {
            std::vector<std::future<int>> results;
            results.reserve(testCases.size());

            // Run each test case in its own thread
            for (auto&  testCase : testCases)
            {
                testCase.values.insert(testCase.values.begin(), executableName);
                results.push_back(std::async(std::launch::async, RunCsvTest, std::cref(testCase), std::cref(runtime)));
            }

            // Check results
            for (auto& result : results)
            {
                if (result.get() != EXIT_SUCCESS)
                {
                    return EXIT_FAILURE;
                }
            }
        }
        else
        {
            // Run tests sequentially
            for (auto&  testCase : testCases)
            {
                testCase.values.insert(testCase.values.begin(), executableName);
                if (RunCsvTest(testCase, runtime) != EXIT_SUCCESS)
                {
                    return EXIT_FAILURE;
                }
            }
        }

        return EXIT_SUCCESS;
    }
    else // Run single test
    {
        // Get the preferred order of compute devices.
        std::vector<armnn::Compute> computeDevices = vm["compute"].as<std::vector<armnn::Compute>>();

        // Remove duplicates from the list of compute devices.
        RemoveDuplicateDevices(computeDevices);

        // Check that the specified compute devices are valid.
        if (!CheckDevicesAreValid(computeDevices))
        {
            BOOST_LOG_TRIVIAL(fatal) << "The list of preferred devices contains an invalid compute";
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

        return RunTest(modelFormat, inputTensorShapeStr, computeDevices,
                       modelPath, inputName, inputTensorDataFilePath, outputName, enableProfiling, subgraphId);
    }
}
