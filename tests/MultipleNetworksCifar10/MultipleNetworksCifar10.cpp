//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
        // Configure logging for both the ARMNN library and this test program
        armnn::ConfigureLogging(true, true, level);
        armnnUtils::ConfigureLogging(boost::log::core::get().get(), true, true, level);

        namespace po = boost::program_options;

        armnn::Compute computeDevice;
        std::string modelDir;
        std::string dataDir;

        po::options_description desc("Options");
        try
        {
            // Add generic options needed for all inference tests
            desc.add_options()
                ("help", "Display help messages")
                ("model-dir,m", po::value<std::string>(&modelDir)->required(),
                    "Path to directory containing the Cifar10 model file")
                ("compute,c", po::value<armnn::Compute>(&computeDevice)->default_value(armnn::Compute::CpuAcc),
                    "Which device to run layers on by default. Possible choices: CpuAcc, CpuRef, GpuAcc")
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

        // Create runtime
        armnn::IRuntimePtr runtime(armnn::IRuntime::Create(computeDevice));

        // Load networks
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
            // Create a network from a file on disk
            armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(modelPath.c_str(), {}, { "prob" });

            // optimize the network
            armnn::IOptimizedNetworkPtr optimizedNet(nullptr, nullptr);
            try
            {
                optimizedNet = armnn::Optimize(*network, runtime->GetDeviceSpec());
            }
            catch (armnn::Exception& e)
            {
                std::stringstream message;
                message << "armnn::Exception ("<<e.what()<<") caught from optimize.";
                BOOST_LOG_TRIVIAL(fatal) << message.str();
                return 1;
            }

            // Load the network into the runtime
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

        // Load a test case and test inference
        if (!ValidateDirectory(dataDir))
        {
            return 1;
        }
        Cifar10Database cifar10(dataDir);

        for (unsigned int i = 0; i < 3; ++i)
        {
            // Load test case data (including image data)
            std::unique_ptr<Cifar10Database::TTestCaseData> testCaseData = cifar10.GetTestCaseData(i);

            // Test inference
            std::vector<std::array<float, 10>> outputs(networksCount);

            for (unsigned int k = 0; k < networksCount; ++k)
            {
                status = runtime->EnqueueWorkload(networks[k].m_Network,
                    MakeInputTensors(networks[k].m_InputBindingInfo, testCaseData->m_InputImage),
                    MakeOutputTensors(networks[k].m_OutputBindingInfo, outputs[k]));
                if (status == armnn::Status::Failure)
                {
                    BOOST_LOG_TRIVIAL(fatal) << "armnn::IRuntime: Failed to enqueue workload";
                    return 1;
                }
            }

            // Compare outputs
            for (unsigned int k = 1; k < networksCount; ++k)
            {
                if (!std::equal(outputs[0].begin(), outputs[0].end(), outputs[k].begin(), outputs[k].end()))
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
