//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <iostream>
#include <chrono>
#include <vector>
#include <array>
#include <boost/log/trivial.hpp>

#include "armnn/ArmNN.hpp"
#include "armnn/Utils.hpp"
#include "armnn/INetwork.hpp"
#include "armnnCaffeParser/ICaffeParser.hpp"
#include "../Cifar10Database.hpp"
#include "../InferenceTest.hpp"
#include "../InferenceModel.hpp"

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
        armnnUtils::ConfigureLogging(boost::log::core::get().get(), true, true, level);

        namespace po = boost::program_options;

        std::vector<armnn::BackendId> computeDevice;
        std::vector<armnn::BackendId> defaultBackends = {armnn::Compute::CpuAcc, armnn::Compute::CpuRef};
        std::string modelDir;
        std::string dataDir;

        const std::string backendsMessage = "Which device to run layers on by default. Possible choices: "
                                          + armnn::BackendRegistryInstance().GetBackendIdsAsString();

        po::options_description desc("Options");
        try
        {
            // Adds generic options needed for all inference tests.
            desc.add_options()
                ("help", "Display help messages")
                ("model-dir,m", po::value<std::string>(&modelDir)->required(),
                    "Path to directory containing the Cifar10 model file")
                ("compute,c", po::value<std::vector<armnn::BackendId>>(&computeDevice)->default_value(defaultBackends),
                    backendsMessage.c_str())
                ("data-dir,d", po::value<std::string>(&dataDir)->required(),
                    "Path to directory containing the Cifar10 test data");
        }
        catch (const std::exception& e)
        {
            // Coverity points out that default_value(...) can throw a bad_lexical_cast,
            // and that desc.add_options() can throw boost::io::too_few_args.
            // They really won't in any of these cases.
            BOOST_ASSERT_MSG(false, "Caught unexpected exception");
            std::cerr << "Fatal internal error: " << e.what() << std::endl;
            return 1;
        }

        po::variables_map vm;

        try
        {
            po::store(po::parse_command_line(argc, argv, desc), vm);

            if (vm.count("help"))
            {
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

        if (!ValidateDirectory(modelDir))
        {
            return 1;
        }
        string modelPath = modelDir + "cifar10_full_iter_60000.caffemodel";

        // Check if the requested backend are all valid
        std::string invalidBackends;
        if (!CheckRequestedBackendsAreValid(computeDevice, armnn::Optional<std::string&>(invalidBackends)))
        {
            BOOST_LOG_TRIVIAL(fatal) << "The list of preferred devices contains invalid backend IDs: "
                                     << invalidBackends;
            return EXIT_FAILURE;
        }

        // Create runtime
        armnn::IRuntime::CreationOptions options;
        armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

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

        armnnCaffeParser::ICaffeParserPtr parser(armnnCaffeParser::ICaffeParser::Create());

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
            catch (armnn::Exception& e)
            {
                std::stringstream message;
                message << "armnn::Exception ("<<e.what()<<") caught from optimize.";
                BOOST_LOG_TRIVIAL(fatal) << message.str();
                return 1;
            }

            // Loads the network into the runtime.
            armnn::NetworkId networkId;
            status = runtime->LoadNetwork(networkId, std::move(optimizedNet));
            if (status == armnn::Status::Failure)
            {
                BOOST_LOG_TRIVIAL(fatal) << "armnn::IRuntime: Failed to load network";
                return 1;
            }

            networks.emplace_back(networkId,
                parser->GetNetworkInputBindingInfo("data"),
                parser->GetNetworkOutputBindingInfo("prob"));
        }

        // Loads a test case and tests inference.
        if (!ValidateDirectory(dataDir))
        {
            return 1;
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
                    BOOST_LOG_TRIVIAL(fatal) << "armnn::IRuntime: Failed to enqueue workload";
                    return 1;
                }
            }

            // Compares outputs.
            std::vector<float> output0 = boost::get<std::vector<float>>(outputs[0]);

            for (unsigned int k = 1; k < networksCount; ++k)
            {
                std::vector<float> outputK = boost::get<std::vector<float>>(outputs[k]);

                if (!std::equal(output0.begin(), output0.end(), outputK.begin(), outputK.end()))
                {
                    BOOST_LOG_TRIVIAL(error) << "Multiple networks inference failed!";
                    return 1;
                }
            }
        }

        BOOST_LOG_TRIVIAL(info) << "Multiple networks inference ran successfully!";
        return 0;
    }
    catch (armnn::Exception const& e)
    {
        // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "Armnn Error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e)
    {
        // Coverity fix: various boost exceptions can be thrown by methods called by this test.
        std::cerr << "WARNING: MultipleNetworksCifar10: An error has occurred when running the "
                     "multiple networks inference tests: " << e.what() << std::endl;
        return 1;
    }
}
