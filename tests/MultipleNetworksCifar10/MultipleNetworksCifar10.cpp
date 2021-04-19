//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnn/ArmNN.hpp"
#include "armnn/Utils.hpp"
#include "armnn/INetwork.hpp"
#include "armnnTfLiteParser/TfLiteParser.hpp"
#include "../Cifar10Database.hpp"
#include "../InferenceTest.hpp"
#include "../InferenceModel.hpp"

#include <cxxopts/cxxopts.hpp>

#include <iostream>
#include <chrono>
#include <vector>
#include <array>


using namespace std;
using namespace std::chrono;
using namespace armnn::test;

int main(int argc, char* argv[])
{
#ifdef NDEBUG
    armnn::LogSeverity level = armnn::LogSeverity::Info;
#else
    armnn::LogSeverity level = armnn::LogSeverity::Debug;
#endif

    try
    {
        // Configures logging for both the ARMNN library and this test program.
        armnn::ConfigureLogging(true, true, level);

        std::vector<armnn::BackendId> computeDevice;
        std::string modelDir;
        std::string dataDir;

        const std::string backendsMessage = "Which device to run layers on by default. Possible choices: "
                                          + armnn::BackendRegistryInstance().GetBackendIdsAsString();

        cxxopts::Options in_options("MultipleNetworksCifar10",
                                    "Run multiple networks inference tests using Cifar-10 data.");

        try
        {
            // Adds generic options needed for all inference tests.
            in_options.add_options()
                ("h,help", "Display help messages")
                ("m,model-dir", "Path to directory containing the Cifar10 model file",
                 cxxopts::value<std::string>(modelDir))
                ("c,compute", backendsMessage.c_str(),
                 cxxopts::value<std::vector<armnn::BackendId>>(computeDevice)->default_value("CpuAcc,CpuRef"))
                ("d,data-dir", "Path to directory containing the Cifar10 test data",
                 cxxopts::value<std::string>(dataDir));

            auto result = in_options.parse(argc, argv);

            if(result.count("help") > 0)
            {
                std::cout << in_options.help() << std::endl;
                return EXIT_FAILURE;
            }

            //ensure mandatory parameters given
            std::string mandatorySingleParameters[] = {"model-dir", "data-dir"};
            for (auto param : mandatorySingleParameters)
            {
                if(result.count(param) > 0)
                {
                    std::string dir = result[param].as<std::string>();

                    if(!ValidateDirectory(dir)) {
                        return EXIT_FAILURE;
                    }
                } else {
                    std::cerr << "Parameter \'--" << param << "\' is required but missing." << std::endl;
                    return EXIT_FAILURE;
                }
            }
        }
        catch (const cxxopts::OptionException& e)
        {
            std::cerr << e.what() << std::endl << in_options.help() << std::endl;
            return EXIT_FAILURE;
        }

        fs::path modelPath = fs::path(modelDir + "/cifar10_tf.prototxt");

        // Create runtime
        // This will also load dynamic backend in case that the dynamic backend path is specified
        armnn::IRuntime::CreationOptions options;
        armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

        // Check if the requested backend are all valid
        std::string invalidBackends;
        if (!CheckRequestedBackendsAreValid(computeDevice, armnn::Optional<std::string&>(invalidBackends)))
        {
            ARMNN_LOG(fatal) << "The list of preferred devices contains invalid backend IDs: "
                             << invalidBackends;
            return EXIT_FAILURE;
        }

        // Loads networks.
        armnn::Status status;
        struct Net
        {
            Net(armnn::NetworkId netId,
                const std::pair<armnn::LayerBindingId, armnn::TensorInfo>& in,
                const std::pair<armnn::LayerBindingId, armnn::TensorInfo>& out)
            : m_Network(netId)
            , m_InputBindingInfo(in)
            , m_OutputBindingInfo(out)
            {}

            armnn::NetworkId m_Network;
            std::pair<armnn::LayerBindingId, armnn::TensorInfo> m_InputBindingInfo;
            std::pair<armnn::LayerBindingId, armnn::TensorInfo> m_OutputBindingInfo;
        };
        std::vector<Net> networks;

        armnnTfLiteParser::ITfLiteParserPtr parser(armnnTfLiteParser::ITfLiteParserPtr::Create());

        const int networksCount = 4;
        for (int i = 0; i < networksCount; ++i)
        {
            // Creates a network from a file on the disk.
            armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(modelPath.c_str(), {}, { "prob" });

            // Optimizes the network.
            armnn::IOptimizedNetworkPtr optimizedNet(nullptr, nullptr);
            try
            {
                optimizedNet = armnn::Optimize(*network, computeDevice, runtime->GetDeviceSpec());
            }
            catch (const armnn::Exception& e)
            {
                std::stringstream message;
                message << "armnn::Exception ("<<e.what()<<") caught from optimize.";
                ARMNN_LOG(fatal) << message.str();
                return EXIT_FAILURE;
            }

            // Loads the network into the runtime.
            armnn::NetworkId networkId;
            status = runtime->LoadNetwork(networkId, std::move(optimizedNet));
            if (status == armnn::Status::Failure)
            {
                ARMNN_LOG(fatal) << "armnn::IRuntime: Failed to load network";
                return EXIT_FAILURE;
            }

            networks.emplace_back(networkId,
                parser->GetNetworkInputBindingInfo("data"),
                parser->GetNetworkOutputBindingInfo("prob"));
        }

        // Loads a test case and tests inference.
        if (!ValidateDirectory(dataDir))
        {
            return EXIT_FAILURE;
        }
        Cifar10Database cifar10(dataDir);

        for (unsigned int i = 0; i < 3; ++i)
        {
            // Loads test case data (including image data).
            std::unique_ptr<Cifar10Database::TTestCaseData> testCaseData = cifar10.GetTestCaseData(i);

            // Tests inference.
            std::vector<TContainer> outputs;
            outputs.reserve(networksCount);

            for (unsigned int j = 0; j < networksCount; ++j)
            {
                outputs.push_back(std::vector<float>(10));
            }

            for (unsigned int k = 0; k < networksCount; ++k)
            {
                std::vector<armnn::BindingPointInfo> inputBindings  = { networks[k].m_InputBindingInfo  };
                std::vector<armnn::BindingPointInfo> outputBindings = { networks[k].m_OutputBindingInfo };

                std::vector<TContainer> inputDataContainers = { testCaseData->m_InputImage };
                std::vector<TContainer> outputDataContainers = { outputs[k] };

                status = runtime->EnqueueWorkload(networks[k].m_Network,
                    armnnUtils::MakeInputTensors(inputBindings, inputDataContainers),
                    armnnUtils::MakeOutputTensors(outputBindings, outputDataContainers));
                if (status == armnn::Status::Failure)
                {
                    ARMNN_LOG(fatal) << "armnn::IRuntime: Failed to enqueue workload";
                    return EXIT_FAILURE;
                }
            }

            // Compares outputs.
            std::vector<float> output0 = mapbox::util::get<std::vector<float>>(outputs[0]);

            for (unsigned int k = 1; k < networksCount; ++k)
            {
                std::vector<float> outputK = mapbox::util::get<std::vector<float>>(outputs[k]);

                if (!std::equal(output0.begin(), output0.end(), outputK.begin(), outputK.end()))
                {
                    ARMNN_LOG(error) << "Multiple networks inference failed!";
                    return EXIT_FAILURE;
                }
            }
        }

        ARMNN_LOG(info) << "Multiple networks inference ran successfully!";
        return EXIT_SUCCESS;
    }
    catch (const armnn::Exception& e)
    {
        // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "Armnn Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (const std::exception& e)
    {
        // Coverity fix: various boost exceptions can be thrown by methods called by this test.
        std::cerr << "WARNING: MultipleNetworksCifar10: An error has occurred when running the "
                     "multiple networks inference tests: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
