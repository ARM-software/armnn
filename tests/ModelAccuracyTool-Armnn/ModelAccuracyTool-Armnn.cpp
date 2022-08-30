//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../ImageTensorGenerator/ImageTensorGenerator.hpp"
#include "../InferenceTest.hpp"
#include "ModelAccuracyChecker.hpp"
#include "armnnDeserializer/IDeserializer.hpp"

#include <armnnUtils/Filesystem.hpp>
#include <armnnUtils/TContainer.hpp>

#include <cxxopts/cxxopts.hpp>
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
 * @param[in] excludelistPath     Path to excludelist file
 * @return A map mapping image file names to their corresponding ground-truth labels
 */
map<std::string, std::string> LoadValidationImageFilenamesAndLabels(const string& validationLabelPath,
                                                                    const string& imageDirectoryPath,
                                                                    size_t begIndex             = 0,
                                                                    size_t endIndex             = 0,
                                                                    const string& excludelistPath = "");

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
        armnn::LogSeverity level = armnn::LogSeverity::Debug;
        armnn::ConfigureLogging(true, true, level);

        std::string modelPath;
        std::string modelFormat;
        std::vector<std::string> inputNames;
        std::vector<std::string> outputNames;
        std::string dataDir;
        std::string modelOutputLabelsPath;
        std::string validationLabelPath;
        std::string inputLayout;
        std::vector<armnn::BackendId> computeDevice;
        std::string validationRange;
        std::string excludelistPath;

        const std::string backendsMessage = "Which device to run layers on by default. Possible choices: "
                                            + armnn::BackendRegistryInstance().GetBackendIdsAsString();

        try
        {
            cxxopts::Options options("ModeAccuracyTool-Armnn","Options");

            options.add_options()
                ("h,help", "Display help messages")
                ("m,model-path",
                    "Path to armnn format model file",
                    cxxopts::value<std::string>(modelPath))
                ("f,model-format",
                    "The model format. Supported values: tflite",
                    cxxopts::value<std::string>(modelFormat))
                ("i,input-name",
                    "Identifier of the input tensors in the network separated by comma with no space.",
                    cxxopts::value<std::vector<std::string>>(inputNames))
                ("o,output-name",
                    "Identifier of the output tensors in the network separated by comma with no space.",
                    cxxopts::value<std::vector<std::string>>(outputNames))
                ("d,data-dir",
                    "Path to directory containing the ImageNet test data",
                    cxxopts::value<std::string>(dataDir))
                ("p,model-output-labels",
                    "Path to model output labels file.",
                    cxxopts::value<std::string>(modelOutputLabelsPath))
                ("v,validation-labels-path",
                    "Path to ImageNet Validation Label file",
                    cxxopts::value<std::string>(validationLabelPath))
                ("l,data-layout",
                    "Data layout. Supported value: NHWC, NCHW. Default: NHWC",
                    cxxopts::value<std::string>(inputLayout)->default_value("NHWC"))
                ("c,compute",
                    backendsMessage.c_str(),
                    cxxopts::value<std::vector<armnn::BackendId>>(computeDevice)->default_value("CpuAcc,CpuRef"))
                ("r,validation-range",
                    "The range of the images to be evaluated. Specified in the form <begin index>:<end index>."
                    "The index starts at 1 and the range is inclusive."
                    "By default the evaluation will be performed on all images.",
                    cxxopts::value<std::string>(validationRange)->default_value("1:0"))
                ("e,excludelist-path",
                    "Path to a excludelist file where each line denotes the index of an image to be "
                    "excluded from evaluation.",
                    cxxopts::value<std::string>(excludelistPath)->default_value(""));

            auto result = options.parse(argc, argv);

            if (result.count("help") > 0)
            {
                std::cout << options.help() << std::endl;
                return EXIT_FAILURE;
            }

            // Check for mandatory single options.
            std::string mandatorySingleParameters[] = { "model-path", "model-format", "input-name", "output-name",
                                                        "data-dir", "model-output-labels", "validation-labels-path" };
            for (auto param : mandatorySingleParameters)
            {
                if (result.count(param) != 1)
                {
                    std::cerr << "Parameter \'--" << param << "\' is required but missing." << std::endl;
                    return EXIT_FAILURE;
                }
            }
        }
        catch (const cxxopts::OptionException& e)
        {
            std::cerr << e.what() << std::endl << std::endl;
            return EXIT_FAILURE;
        }
        catch (const std::exception& e)
        {
            ARMNN_ASSERT_MSG(false, "Caught unexpected exception");
            std::cerr << "Fatal internal error: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }

        // Check if the requested backend are all valid
        std::string invalidBackends;
        if (!CheckRequestedBackendsAreValid(computeDevice, armnn::Optional<std::string&>(invalidBackends)))
        {
            ARMNN_LOG(fatal) << "The list of preferred devices contains invalid backend IDs: "
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
        catch (const armnn::Exception& e)
        {
            std::stringstream message;
            message << "armnn::Exception (" << e.what() << ") caught from optimize.";
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

        // Set up Network
        using BindingPointInfo = InferenceModelInternal::BindingPointInfo;

        // Handle inputNames and outputNames, there can be multiple.
        std::vector<BindingPointInfo> inputBindings;
        for(auto& input: inputNames)
        {
            const armnnDeserializer::BindingPointInfo&
                    inputBindingInfo = armnnparser->GetNetworkInputBindingInfo(0, input);

            std::pair<armnn::LayerBindingId, armnn::TensorInfo>
                    m_InputBindingInfo(inputBindingInfo.m_BindingId, inputBindingInfo.m_TensorInfo);
            inputBindings.push_back(m_InputBindingInfo);
        }

        std::vector<BindingPointInfo> outputBindings;
        for(auto& output: outputNames)
        {
            const armnnDeserializer::BindingPointInfo&
                    outputBindingInfo = armnnparser->GetNetworkOutputBindingInfo(0, output);

            std::pair<armnn::LayerBindingId, armnn::TensorInfo>
                    m_OutputBindingInfo(outputBindingInfo.m_BindingId, outputBindingInfo.m_TensorInfo);
            outputBindings.push_back(m_OutputBindingInfo);
        }

        // Load model output labels
        if (modelOutputLabelsPath.empty() || !fs::exists(modelOutputLabelsPath) ||
            !fs::is_regular_file(modelOutputLabelsPath))
        {
            ARMNN_LOG(fatal) << "Invalid model output labels path at " << modelOutputLabelsPath;
        }
        const std::vector<armnnUtils::LabelCategoryNames> modelOutputLabels =
            LoadModelOutputLabels(modelOutputLabelsPath);

        // Parse begin and end image indices
        std::vector<std::string> imageIndexStrs = armnnUtils::SplitBy(validationRange, ":");
        size_t imageBegIndex;
        size_t imageEndIndex;
        if (imageIndexStrs.size() != 2)
        {
            ARMNN_LOG(fatal) << "Invalid validation range specification: Invalid format " << validationRange;
            return EXIT_FAILURE;
        }
        try
        {
            imageBegIndex = std::stoul(imageIndexStrs[0]);
            imageEndIndex = std::stoul(imageIndexStrs[1]);
        }
        catch (const std::exception& e)
        {
            ARMNN_LOG(fatal) << "Invalid validation range specification: " << validationRange;
            return EXIT_FAILURE;
        }

        // Validate  excludelist file if it's specified
        if (!excludelistPath.empty() &&
            !(fs::exists(excludelistPath) && fs::is_regular_file(excludelistPath)))
        {
            ARMNN_LOG(fatal) << "Invalid path to excludelist file at " << excludelistPath;
            return EXIT_FAILURE;
        }

        fs::path pathToDataDir(dataDir);
        const map<std::string, std::string> imageNameToLabel = LoadValidationImageFilenamesAndLabels(
            validationLabelPath, pathToDataDir.string(), imageBegIndex, imageEndIndex, excludelistPath);
        armnnUtils::ModelAccuracyChecker checker(imageNameToLabel, modelOutputLabels);

        if (ValidateDirectory(dataDir))
        {
            InferenceModel<armnnDeserializer::IDeserializer, float>::Params params;

            params.m_ModelPath      = modelPath;
            params.m_IsModelBinary  = true;
            params.m_ComputeDevices = computeDevice;
            // Insert inputNames and outputNames into params vector
            params.m_InputBindings.insert(std::end(params.m_InputBindings),
                                          std::begin(inputNames),
                                          std::end(inputNames));
            params.m_OutputBindings.insert(std::end(params.m_OutputBindings),
                                           std::begin(outputNames),
                                           std::end(outputNames));

            using TParser = armnnDeserializer::IDeserializer;
            // If dynamicBackends is empty it will be disabled by default.
            InferenceModel<TParser, float> model(params, false, "");

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
                ARMNN_LOG(fatal) << "Invalid Data layout: " << inputLayout;
                return EXIT_FAILURE;
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
                ARMNN_LOG(fatal) << "Number of output elements: " << outputNumElements
                                         << " , mismatches the number of output labels: " << modelOutputLabels.size();
                return EXIT_FAILURE;
            }

            const unsigned int batchSize = 1;
            // Get normalisation parameters
            SupportedFrontend modelFrontend;
            if (modelFormat == "tflite")
            {
                modelFrontend = SupportedFrontend::TFLite;
            }
            else
            {
                ARMNN_LOG(fatal) << "Unsupported frontend: " << modelFormat;
                return EXIT_FAILURE;
            }
            const NormalizationParameters& normParams = GetNormalizationParameters(modelFrontend, inputTensorDataType);
            for (const auto& imageEntry : imageNameToLabel)
            {
                const std::string imageName = imageEntry.first;
                std::cout << "Processing image: " << imageName << "\n";

                vector<armnnUtils::TContainer> inputDataContainers;
                vector<armnnUtils::TContainer> outputDataContainers;

                auto imagePath = pathToDataDir / fs::path(imageName);
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
                    case armnn::DataType::QAsymmU8:
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
                    ARMNN_LOG(fatal) << "armnn::IRuntime: Failed to enqueue workload for image: " << imageName;
                }

                checker.AddImageResult<armnnUtils::TContainer>(imageName, outputDataContainers);
            }
        }
        else
        {
            return EXIT_SUCCESS;
        }

        for(unsigned int i = 1; i <= 5; ++i)
        {
            std::cout << "Top " << i <<  " Accuracy: " << checker.GetAccuracy(i) << "%" << "\n";
        }

        ARMNN_LOG(info) << "Accuracy Tool ran successfully!";
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
        std::cerr << "WARNING: ModelAccuracyTool-Armnn: An error has occurred when running the "
                     "Accuracy Tool: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

map<std::string, std::string> LoadValidationImageFilenamesAndLabels(const string& validationLabelPath,
                                                                    const string& imageDirectoryPath,
                                                                    size_t begIndex,
                                                                    size_t endIndex,
                                                                    const string& excludelistPath)
{
    // Populate imageFilenames with names of all .JPEG, .PNG images
    std::vector<std::string> imageFilenames;
    for (const auto& imageEntry : fs::directory_iterator(fs::path(imageDirectoryPath)))
    {
        fs::path imagePath = imageEntry.path();

        // Get extension and convert to uppercase
        std::string imageExtension = imagePath.extension().string();
        std::transform(imageExtension.begin(), imageExtension.end(), imageExtension.begin(), ::toupper);

        if (fs::is_regular_file(imagePath) && (imageExtension == ".JPEG" || imageExtension == ".PNG"))
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

    // Load excludelist if there is one
    std::vector<unsigned int> excludelist;
    if (!excludelistPath.empty())
    {
        std::ifstream excludelistFile(excludelistPath);
        unsigned int index;
        while (excludelistFile >> index)
        {
            excludelist.push_back(index);
        }
    }

    // Load ground truth labels and pair them with corresponding image names
    std::string classification;
    map<std::string, std::string> imageNameToLabel;
    ifstream infile(validationLabelPath);
    size_t imageIndex          = begIndex;
    size_t excludelistIndexCount = 0;
    while (std::getline(infile, classification))
    {
        if (imageIndex > endIndex)
        {
            break;
        }
        // If current imageIndex is included in excludelist, skip the current image
        if (excludelistIndexCount < excludelist.size() && imageIndex == excludelist[excludelistIndexCount])
        {
            ++imageIndex;
            ++excludelistIndexCount;
            continue;
        }
        imageNameToLabel.insert(std::pair<std::string, std::string>(imageFilenames[imageIndex - 1], classification));
        ++imageIndex;
    }
    std::cout << excludelistIndexCount << " images in excludelist" << std::endl;
    std::cout << imageIndex - begIndex - excludelistIndexCount << " images to be loaded" << std::endl;
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