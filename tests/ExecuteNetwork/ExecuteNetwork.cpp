//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../NetworkExecutionUtils/NetworkExecutionUtils.hpp"

// MAIN
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
    std::string inputNames;
    std::string inputTensorShapes;
    std::string inputTensorDataFilePaths;
    std::string outputNames;
    std::string inputTypes;
    std::string outputTypes;
    std::string dynamicBackendsPath;

    double thresholdTime = 0.0;

    size_t subgraphId = 0;

    const std::string backendsMessage = "Which device to run layers on by default. Possible choices: "
                                      + armnn::BackendRegistryInstance().GetBackendIdsAsString();

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
            ("model-format,f", po::value(&modelFormat)->required(),
             "armnn-binary, caffe-binary, caffe-text, onnx-binary, onnx-text, tflite-binary, tensorflow-binary or "
             "tensorflow-text.")
            ("model-path,m", po::value(&modelPath)->required(), "Path to model file, e.g. .armnn, .caffemodel, "
             ".prototxt, .tflite, .onnx")
            ("compute,c", po::value<std::vector<std::string>>()->multitoken(),
             backendsMessage.c_str())
            ("dynamic-backends-path,b", po::value(&dynamicBackendsPath),
             "Path where to load any available dynamic backend from. "
             "If left empty (the default), dynamic backends will not be used.")
            ("input-name,i", po::value(&inputNames),
             "Identifier of the input tensors in the network separated by comma.")
            ("subgraph-number,x", po::value<size_t>(&subgraphId)->default_value(0), "Id of the subgraph to be executed."
              "Defaults to 0")
            ("input-tensor-shape,s", po::value(&inputTensorShapes),
             "The shape of the input tensors in the network as a flat array of integers separated by comma. "
             "Several shapes can be passed separating them by semicolon. "
             "This parameter is optional, depending on the network.")
            ("input-tensor-data,d", po::value(&inputTensorDataFilePaths),
             "Path to files containing the input data as a flat array separated by whitespace. "
             "Several paths can be passed separating them by comma.")
            ("input-type,y",po::value(&inputTypes), "The type of the input tensors in the network separated by comma. "
             "If unset, defaults to \"float\" for all defined inputs. "
             "Accepted values (float, int or qasymm8)")
            ("quantize-input,q",po::bool_switch()->default_value(false),
             "If this option is enabled, all float inputs will be quantized to qasymm8. "
             "If unset, default to not quantized. "
             "Accepted values (true or false)")
            ("output-type,z",po::value(&outputTypes),
             "The type of the output tensors in the network separated by comma. "
             "If unset, defaults to \"float\" for all defined outputs. "
             "Accepted values (float, int or qasymm8).")
            ("output-name,o", po::value(&outputNames),
             "Identifier of the output tensors in the network separated by comma.")
            ("event-based-profiling,e", po::bool_switch()->default_value(false),
             "Enables built in profiler. If unset, defaults to off.")
            ("fp16-turbo-mode,h", po::bool_switch()->default_value(false), "If this option is enabled, FP32 layers, "
             "weights and biases will be converted to FP16 where the backend supports it")
            ("threshold-time,r", po::value<double>(&thresholdTime)->default_value(0.0),
             "Threshold time is the maximum allowed time for inference measured in milliseconds. If the actual "
             "inference time is greater than the threshold time, the test will fail. By default, no threshold "
             "time is used.");
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
    bool enableFp16TurboMode = vm["fp16-turbo-mode"].as<bool>();
    bool quantizeInput = vm["quantize-input"].as<bool>();

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
        options.m_EnableGpuProfiling = enableProfiling;

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
                results.push_back(std::async(std::launch::async, RunCsvTest, std::cref(testCase), std::cref(runtime),
                                             enableProfiling, enableFp16TurboMode, thresholdTime));
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
                if (RunCsvTest(testCase, runtime, enableProfiling, enableFp16TurboMode, thresholdTime) != EXIT_SUCCESS)
                {
                    return EXIT_FAILURE;
                }
            }
        }

        return EXIT_SUCCESS;
    }
    else // Run single test
    {
        // Get the preferred order of compute devices. If none are specified, default to using CpuRef
        const std::string computeOption("compute");
        std::vector<std::string> computeDevicesAsStrings =
                CheckOption(vm, computeOption.c_str()) ?
                    vm[computeOption].as<std::vector<std::string>>() :
                    std::vector<std::string>();
        std::vector<armnn::BackendId> computeDevices(computeDevicesAsStrings.begin(), computeDevicesAsStrings.end());

        // Remove duplicates from the list of compute devices.
        RemoveDuplicateDevices(computeDevices);

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

        return RunTest(modelFormat, inputTensorShapes, computeDevices, dynamicBackendsPath, modelPath, inputNames,
                       inputTensorDataFilePaths, inputTypes, quantizeInput, outputTypes, outputNames,
                       enableProfiling, enableFp16TurboMode, thresholdTime, subgraphId);
    }
}
