//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ModelAccuracyChecker.hpp"
#include "../InferenceTest.hpp"
#include "../ImagePreprocessor.hpp"
#include "armnnDeserializer/IDeserializer.hpp"

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace armnn::test;

namespace po = boost::program_options;

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

map<std::string, int> LoadValidationLabels(const string & validationLabelPath);

template<armnn::DataType NonQuantizedType>
auto ParseDataArray(std::istream & stream);

template<>
auto ParseDataArray<armnn::DataType::Float32>(std::istream & stream)
{
    return ParseArrayImpl<float>(stream, [](const std::string& s) { return std::stof(s); });
}

int main(int argc, char* argv[])
{
    try
    {
        using namespace boost::filesystem;
        armnn::LogSeverity level = armnn::LogSeverity::Debug;
        armnn::ConfigureLogging(true, true, level);
        armnnUtils::ConfigureLogging(boost::log::core::get().get(), true, true, level);

        // Set-up program Options
        namespace po = boost::program_options;

        std::vector<armnn::BackendId> computeDevice;
        std::vector<armnn::BackendId> defaultBackends = {armnn::Compute::CpuAcc, armnn::Compute::CpuRef};
        std::string modelPath;
        std::string dataDir;
        std::string inputName;
        std::string outputName;
        std::string validationLabelPath;

        const std::string backendsMessage = "Which device to run layers on by default. Possible choices: "
                                            + armnn::BackendRegistryInstance().GetBackendIdsAsString();

        po::options_description desc("Options");
        try
        {
            // Adds generic options needed to run Accuracy Tool.
            desc.add_options()
                ("help,h", "Display help messages")
                ("model-path,m", po::value<std::string>(&modelPath)->required(), "Path to armnn format model file")
                ("compute,c", po::value<std::vector<armnn::BackendId>>(&computeDevice)->default_value(defaultBackends),
                 backendsMessage.c_str())
                ("data-dir,d", po::value<std::string>(&dataDir)->required(),
                 "Path to directory containing the ImageNet test data")
                ("input-name,i", po::value<std::string>(&inputName)->required(),
                 "Identifier of the input tensors in the network separated by comma.")
                ("output-name,o", po::value<std::string>(&outputName)->required(),
                 "Identifier of the output tensors in the network separated by comma.")
                ("validation-labels-path,v", po::value<std::string>(&validationLabelPath)->required(),
                 "Path to ImageNet Validation Label file");
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

        // Check if the requested backend are all valid
        std::string invalidBackends;
        if (!CheckRequestedBackendsAreValid(computeDevice, armnn::Optional<std::string&>(invalidBackends)))
        {
            BOOST_LOG_TRIVIAL(fatal) << "The list of preferred devices contains invalid backend IDs: "
                                     << invalidBackends;
            return EXIT_FAILURE;
        }
        armnn::Status status;

        // Create runtime
        armnn::IRuntime::CreationOptions options;
        armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
        std::ifstream file(modelPath);

        // Create Parser
        using IParser = armnnDeserializer::IDeserializer;
        auto armnnparser(IParser::Create());

        // Create a network
        armnn::INetworkPtr network = armnnparser->CreateNetworkFromBinary(file);

        // Optimizes the network.
        armnn::IOptimizedNetworkPtr optimizedNet(nullptr, nullptr);
        try
        {
            optimizedNet = armnn::Optimize(*network, computeDevice, runtime->GetDeviceSpec());
        }
        catch (armnn::Exception& e)
        {
            std::stringstream message;
            message << "armnn::Exception (" << e.what() << ") caught from optimize.";
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

        // Set up Network
        using BindingPointInfo = InferenceModelInternal::BindingPointInfo;

        const armnnDeserializer::BindingPointInfo&
            inputBindingInfo = armnnparser->GetNetworkInputBindingInfo(0, inputName);

        std::pair<armnn::LayerBindingId, armnn::TensorInfo>
            m_InputBindingInfo(inputBindingInfo.m_BindingId, inputBindingInfo.m_TensorInfo);
        std::vector<BindingPointInfo> inputBindings  = { m_InputBindingInfo };

        const armnnDeserializer::BindingPointInfo&
            outputBindingInfo = armnnparser->GetNetworkOutputBindingInfo(0, outputName);

        std::pair<armnn::LayerBindingId, armnn::TensorInfo>
            m_OutputBindingInfo(outputBindingInfo.m_BindingId, outputBindingInfo.m_TensorInfo);
        std::vector<BindingPointInfo> outputBindings = { m_OutputBindingInfo };

        path pathToDataDir(dataDir);
        map<string, int> validationLabels = LoadValidationLabels(validationLabelPath);
        armnnUtils::ModelAccuracyChecker checker(validationLabels);
        using TContainer = boost::variant<std::vector<float>, std::vector<int>, std::vector<uint8_t>>;

        if(ValidateDirectory(dataDir))
        {
            for (auto & imageEntry : boost::make_iterator_range(directory_iterator(pathToDataDir), {}))
            {
                cout << "Processing image: " << imageEntry << "\n";

                std::ifstream inputTensorFile(imageEntry.path().string());
                vector<TContainer> inputDataContainers;
                inputDataContainers.push_back(ParseDataArray<armnn::DataType::Float32>(inputTensorFile));
                vector<TContainer> outputDataContainers = {vector<float>(1001)};

                status = runtime->EnqueueWorkload(networkId,
                                                  armnnUtils::MakeInputTensors(inputBindings, inputDataContainers),
                                                  armnnUtils::MakeOutputTensors(outputBindings, outputDataContainers));

                if (status == armnn::Status::Failure)
                {
                    BOOST_LOG_TRIVIAL(fatal) << "armnn::IRuntime: Failed to enqueue workload for image: " << imageEntry;
                }

                const std::string imageName = imageEntry.path().filename().string();
                checker.AddImageResult<TContainer>(imageName, outputDataContainers);
            }
        }
        else
        {
            return 1;
        }

        for(unsigned int i = 1; i <= 5; ++i)
        {
            std::cout << "Top " << i <<  " Accuracy: " << checker.GetAccuracy(i) << "%" << "\n";
        }

        BOOST_LOG_TRIVIAL(info) << "Accuracy Tool ran successfully!";
        return 0;
    }
    catch (armnn::Exception const & e)
    {
        // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "Armnn Error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception & e)
    {
        // Coverity fix: various boost exceptions can be thrown by methods called by this test.
        std::cerr << "WARNING: ModelAccuracyTool-Armnn: An error has occurred when running the "
                     "Accuracy Tool: " << e.what() << std::endl;
        return 1;
    }
}

map<std::string, int> LoadValidationLabels(const string & validationLabelPath)
{
    std::string imageName;
    int classification;
    map<std::string, int> validationLabel;
    ifstream infile(validationLabelPath);
    while (infile >> imageName >> classification)
    {
        std::string trimmedName;
        size_t lastindex = imageName.find_last_of(".");
        if(lastindex != std::string::npos)
        {
            trimmedName = imageName.substr(0, lastindex);
        }
        validationLabel.insert(pair<string, int>(trimmedName, classification));
    }
    return validationLabel;
}
