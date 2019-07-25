//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../ImageTensorGenerator/ImageTensorGenerator.hpp"
#include "../InferenceTest.hpp"
#include "ModelAccuracyChecker.hpp"
#include "armnnDeserializer/IDeserializer.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/range/iterator_range.hpp>
#include <map>

using namespace armnn::test;

/** Load image names and ground-truth labels from the image directory and the ground truth label file
 *
 * @pre \p validationLabelPath exists and is valid regular file
 * @pre \p imageDirectoryPath exists and is valid directory
 * @pre labels in validation file correspond to images which are in lexicographical order with the image name
 * @pre image index starts at 1
 * @pre \p begIndex and \p endIndex are end-inclusive
 *
 * @param[in] validationLabelPath Path to validation label file
 * @param[in] imageDirectoryPath  Path to directory containing validation images
 * @param[in] begIndex            Begin index of images to be loaded. Inclusive
 * @param[in] endIndex            End index of images to be loaded. Inclusive
 * @param[in] blacklistPath       Path to blacklist file
 * @return A map mapping image file names to their corresponding ground-truth labels
 */
map<std::string, std::string> LoadValidationImageFilenamesAndLabels(const string& validationLabelPath,
                                                                    const string& imageDirectoryPath,
                                                                    size_t begIndex             = 0,
                                                                    size_t endIndex             = 0,
                                                                    const string& blacklistPath = "");

/** Load model output labels from file
 * 
 * @pre \p modelOutputLabelsPath exists and is a regular file
 *
 * @param[in] modelOutputLabelsPath path to model output labels file
 * @return A vector of labels, which in turn is described by a list of category names
 */
std::vector<armnnUtils::LabelCategoryNames> LoadModelOutputLabels(const std::string& modelOutputLabelsPath);

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
        std::string modelFormat;
        std::string dataDir;
        std::string inputName;
        std::string inputLayout;
        std::string outputName;
        std::string modelOutputLabelsPath;
        std::string validationLabelPath;
        std::string validationRange;
        std::string blacklistPath;

        const std::string backendsMessage = "Which device to run layers on by default. Possible choices: "
                                            + armnn::BackendRegistryInstance().GetBackendIdsAsString();

        po::options_description desc("Options");
        try
        {
            // Adds generic options needed to run Accuracy Tool.
            desc.add_options()
                ("help,h", "Display help messages")
                ("model-path,m", po::value<std::string>(&modelPath)->required(), "Path to armnn format model file")
                ("model-format,f", po::value<std::string>(&modelFormat)->required(),
                 "The model format. Supported values: caffe, tensorflow, tflite")
                ("input-name,i", po::value<std::string>(&inputName)->required(),
                 "Identifier of the input tensors in the network separated by comma.")
                ("output-name,o", po::value<std::string>(&outputName)->required(),
                 "Identifier of the output tensors in the network separated by comma.")
                ("data-dir,d", po::value<std::string>(&dataDir)->required(),
                 "Path to directory containing the ImageNet test data")
                ("model-output-labels,p", po::value<std::string>(&modelOutputLabelsPath)->required(),
                 "Path to model output labels file.")
                ("validation-labels-path,v", po::value<std::string>(&validationLabelPath)->required(),
                 "Path to ImageNet Validation Label file")
                ("data-layout,l", po::value<std::string>(&inputLayout)->default_value("NHWC"),
                 "Data layout. Supported value: NHWC, NCHW. Default: NHWC")
                ("compute,c", po::value<std::vector<armnn::BackendId>>(&computeDevice)->default_value(defaultBackends),
                 backendsMessage.c_str())
                ("validation-range,r", po::value<std::string>(&validationRange)->default_value("1:0"),
                 "The range of the images to be evaluated. Specified in the form <begin index>:<end index>."
                 "The index starts at 1 and the range is inclusive."
                 "By default the evaluation will be performed on all images.")
                ("blacklist-path,b", po::value<std::string>(&blacklistPath)->default_value(""),
                 "Path to a blacklist file where each line denotes the index of an image to be "
                 "excluded from evaluation.");
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

        // Load model output labels
        if (modelOutputLabelsPath.empty() || !boost::filesystem::exists(modelOutputLabelsPath) ||
            !boost::filesystem::is_regular_file(modelOutputLabelsPath))
        {
            BOOST_LOG_TRIVIAL(fatal) << "Invalid model output labels path at " << modelOutputLabelsPath;
        }
        const std::vector<armnnUtils::LabelCategoryNames> modelOutputLabels =
            LoadModelOutputLabels(modelOutputLabelsPath);

        // Parse begin and end image indices
        std::vector<std::string> imageIndexStrs = armnnUtils::SplitBy(validationRange, ":");
        size_t imageBegIndex;
        size_t imageEndIndex;
        if (imageIndexStrs.size() != 2)
        {
            BOOST_LOG_TRIVIAL(fatal) << "Invalid validation range specification: Invalid format " << validationRange;
            return 1;
        }
        try
        {
            imageBegIndex = std::stoul(imageIndexStrs[0]);
            imageEndIndex = std::stoul(imageIndexStrs[1]);
        }
        catch (const std::exception& e)
        {
            BOOST_LOG_TRIVIAL(fatal) << "Invalid validation range specification: " << validationRange;
            return 1;
        }

        // Validate  blacklist file if it's specified
        if (!blacklistPath.empty() &&
            !(boost::filesystem::exists(blacklistPath) && boost::filesystem::is_regular_file(blacklistPath)))
        {
            BOOST_LOG_TRIVIAL(fatal) << "Invalid path to blacklist file at " << blacklistPath;
            return 1;
        }

        path pathToDataDir(dataDir);
        const map<std::string, std::string> imageNameToLabel = LoadValidationImageFilenamesAndLabels(
            validationLabelPath, pathToDataDir.string(), imageBegIndex, imageEndIndex, blacklistPath);
        armnnUtils::ModelAccuracyChecker checker(imageNameToLabel, modelOutputLabels);
        using TContainer = boost::variant<std::vector<float>, std::vector<int>, std::vector<uint8_t>>;

        if (ValidateDirectory(dataDir))
        {
            InferenceModel<armnnDeserializer::IDeserializer, float>::Params params;
            params.m_ModelPath      = modelPath;
            params.m_IsModelBinary  = true;
            params.m_ComputeDevices = computeDevice;
            params.m_InputBindings.push_back(inputName);
            params.m_OutputBindings.push_back(outputName);

            using TParser = armnnDeserializer::IDeserializer;
            InferenceModel<TParser, float> model(params, false);
            // Get input tensor information
            const armnn::TensorInfo& inputTensorInfo   = model.GetInputBindingInfo().second;
            const armnn::TensorShape& inputTensorShape = inputTensorInfo.GetShape();
            const armnn::DataType& inputTensorDataType = inputTensorInfo.GetDataType();
            armnn::DataLayout inputTensorDataLayout;
            if (inputLayout == "NCHW")
            {
                inputTensorDataLayout = armnn::DataLayout::NCHW;
            }
            else if (inputLayout == "NHWC")
            {
                inputTensorDataLayout = armnn::DataLayout::NHWC;
            }
            else
            {
                BOOST_LOG_TRIVIAL(fatal) << "Invalid Data layout: " << inputLayout;
                return 1;
            }
            const unsigned int inputTensorWidth =
                inputTensorDataLayout == armnn::DataLayout::NCHW ? inputTensorShape[3] : inputTensorShape[2];
            const unsigned int inputTensorHeight =
                inputTensorDataLayout == armnn::DataLayout::NCHW ? inputTensorShape[2] : inputTensorShape[1];
            // Get output tensor info
            const unsigned int outputNumElements = model.GetOutputSize();
            // Check output tensor shape is valid
            if (modelOutputLabels.size() != outputNumElements)
            {
                BOOST_LOG_TRIVIAL(fatal) << "Number of output elements: " << outputNumElements
                                         << " , mismatches the number of output labels: " << modelOutputLabels.size();
                return 1;
            }

            const unsigned int batchSize = 1;
            // Get normalisation parameters
            SupportedFrontend modelFrontend;
            if (modelFormat == "caffe")
            {
                modelFrontend = SupportedFrontend::Caffe;
            }
            else if (modelFormat == "tensorflow")
            {
                modelFrontend = SupportedFrontend::TensorFlow;
            }
            else if (modelFormat == "tflite")
            {
                modelFrontend = SupportedFrontend::TFLite;
            }
            else
            {
                BOOST_LOG_TRIVIAL(fatal) << "Unsupported frontend: " << modelFormat;
                return 1;
            }
            const NormalizationParameters& normParams = GetNormalizationParameters(modelFrontend, inputTensorDataType);
            for (const auto& imageEntry : imageNameToLabel)
            {
                const std::string imageName = imageEntry.first;
                std::cout << "Processing image: " << imageName << "\n";

                vector<TContainer> inputDataContainers;
                vector<TContainer> outputDataContainers;

                auto imagePath = pathToDataDir / boost::filesystem::path(imageName);
                switch (inputTensorDataType)
                {
                    case armnn::DataType::Signed32:
                        inputDataContainers.push_back(
                            PrepareImageTensor<int>(imagePath.string(),
                            inputTensorWidth, inputTensorHeight,
                            normParams,
                            batchSize,
                            inputTensorDataLayout));
                        outputDataContainers = { vector<int>(outputNumElements) };
                        break;
                    case armnn::DataType::QuantisedAsymm8:
                        inputDataContainers.push_back(
                            PrepareImageTensor<uint8_t>(imagePath.string(),
                            inputTensorWidth, inputTensorHeight,
                            normParams,
                            batchSize,
                            inputTensorDataLayout));
                        outputDataContainers = { vector<uint8_t>(outputNumElements) };
                        break;
                    case armnn::DataType::Float32:
                    default:
                        inputDataContainers.push_back(
                            PrepareImageTensor<float>(imagePath.string(),
                            inputTensorWidth, inputTensorHeight,
                            normParams,
                            batchSize,
                            inputTensorDataLayout));
                        outputDataContainers = { vector<float>(outputNumElements) };
                        break;
                }

                status = runtime->EnqueueWorkload(networkId,
                                                  armnnUtils::MakeInputTensors(inputBindings, inputDataContainers),
                                                  armnnUtils::MakeOutputTensors(outputBindings, outputDataContainers));

                if (status == armnn::Status::Failure)
                {
                    BOOST_LOG_TRIVIAL(fatal) << "armnn::IRuntime: Failed to enqueue workload for image: " << imageName;
                }

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

map<std::string, std::string> LoadValidationImageFilenamesAndLabels(const string& validationLabelPath,
                                                                    const string& imageDirectoryPath,
                                                                    size_t begIndex,
                                                                    size_t endIndex,
                                                                    const string& blacklistPath)
{
    // Populate imageFilenames with names of all .JPEG, .PNG images
    std::vector<std::string> imageFilenames;
    for (const auto& imageEntry :
         boost::make_iterator_range(boost::filesystem::directory_iterator(boost::filesystem::path(imageDirectoryPath))))
    {
        boost::filesystem::path imagePath = imageEntry.path();
        std::string imageExtension        = boost::to_upper_copy<std::string>(imagePath.extension().string());
        if (boost::filesystem::is_regular_file(imagePath) && (imageExtension == ".JPEG" || imageExtension == ".PNG"))
        {
            imageFilenames.push_back(imagePath.filename().string());
        }
    }
    if (imageFilenames.empty())
    {
        throw armnn::Exception("No image file (JPEG, PNG) found at " + imageDirectoryPath);
    }

    // Sort the image filenames lexicographically
    std::sort(imageFilenames.begin(), imageFilenames.end());

    std::cout << imageFilenames.size() << " images found at " << imageDirectoryPath << std::endl;

    // Get default end index
    if (begIndex < 1 || endIndex > imageFilenames.size())
    {
        throw armnn::Exception("Invalid image index range");
    }
    endIndex = endIndex == 0 ? imageFilenames.size() : endIndex;
    if (begIndex > endIndex)
    {
        throw armnn::Exception("Invalid image index range");
    }

    // Load blacklist if there is one
    std::vector<unsigned int> blacklist;
    if (!blacklistPath.empty())
    {
        std::ifstream blacklistFile(blacklistPath);
        unsigned int index;
        while (blacklistFile >> index)
        {
            blacklist.push_back(index);
        }
    }

    // Load ground truth labels and pair them with corresponding image names
    std::string classification;
    map<std::string, std::string> imageNameToLabel;
    ifstream infile(validationLabelPath);
    size_t imageIndex          = begIndex;
    size_t blacklistIndexCount = 0;
    while (std::getline(infile, classification))
    {
        if (imageIndex > endIndex)
        {
            break;
        }
        // If current imageIndex is included in blacklist, skip the current image
        if (blacklistIndexCount < blacklist.size() && imageIndex == blacklist[blacklistIndexCount])
        {
            ++imageIndex;
            ++blacklistIndexCount;
            continue;
        }
        imageNameToLabel.insert(std::pair<std::string, std::string>(imageFilenames[imageIndex - 1], classification));
        ++imageIndex;
    }
    std::cout << blacklistIndexCount << " images blacklisted" << std::endl;
    std::cout << imageIndex - begIndex - blacklistIndexCount << " images to be loaded" << std::endl;
    return imageNameToLabel;
}

std::vector<armnnUtils::LabelCategoryNames> LoadModelOutputLabels(const std::string& modelOutputLabelsPath)
{
    std::vector<armnnUtils::LabelCategoryNames> modelOutputLabels;
    ifstream modelOutputLablesFile(modelOutputLabelsPath);
    std::string line;
    while (std::getline(modelOutputLablesFile, line))
    {
        armnnUtils::LabelCategoryNames tokens                  = armnnUtils::SplitBy(line, ":");
        armnnUtils::LabelCategoryNames predictionCategoryNames = armnnUtils::SplitBy(tokens.back(), ",");
        std::transform(predictionCategoryNames.begin(), predictionCategoryNames.end(), predictionCategoryNames.begin(),
                       [](const std::string& category) { return armnnUtils::Strip(category); });
        modelOutputLabels.push_back(predictionCategoryNames);
    }
    return modelOutputLabels;
}