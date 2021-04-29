//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnTfLiteParser/ITfLiteParser.hpp"

#include "NMS.hpp"

#include <stb/stb_image.h>

#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Logging.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <cxxopts/cxxopts.hpp>
#include <ghc/filesystem.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <cmath>

using namespace armnnTfLiteParser;
using namespace armnn;

static const int OPEN_FILE_ERROR = -2;
static const int OPTIMIZE_NETWORK_ERROR = -3;
static const int LOAD_NETWORK_ERROR = -4;
static const int LOAD_IMAGE_ERROR = -5;
static const int GENERAL_ERROR = -100;

#define CHECK_OK(v)                                     \
    do {                                                \
        try {                                           \
            auto r_local = v;                           \
            if (r_local != 0) { return r_local;}        \
        }                                               \
        catch (const armnn::Exception& e)               \
        {                                               \
            ARMNN_LOG(error) << "Oops: " << e.what();   \
            return GENERAL_ERROR;                       \
        }                                               \
    } while(0)



template<typename TContainer>
inline armnn::InputTensors MakeInputTensors(const std::vector<armnn::BindingPointInfo>& inputBindings,
                                            const std::vector<std::reference_wrapper<TContainer>>& inputDataContainers)
{
    armnn::InputTensors inputTensors;

    const size_t numInputs = inputBindings.size();
    if (numInputs != inputDataContainers.size())
    {
        throw armnn::Exception("Mismatching vectors");
    }

    for (size_t i = 0; i < numInputs; i++)
    {
        const armnn::BindingPointInfo& inputBinding = inputBindings[i];
        const TContainer& inputData = inputDataContainers[i].get();

        armnn::ConstTensor inputTensor(inputBinding.second, inputData.data());
        inputTensors.push_back(std::make_pair(inputBinding.first, inputTensor));
    }

    return inputTensors;
}

template<typename TContainer>
inline armnn::OutputTensors MakeOutputTensors(
    const std::vector<armnn::BindingPointInfo>& outputBindings,
    const std::vector<std::reference_wrapper<TContainer>>& outputDataContainers)
{
    armnn::OutputTensors outputTensors;

    const size_t numOutputs = outputBindings.size();
    if (numOutputs != outputDataContainers.size())
    {
        throw armnn::Exception("Mismatching vectors");
    }

    outputTensors.reserve(numOutputs);

    for (size_t i = 0; i < numOutputs; i++)
    {
        const armnn::BindingPointInfo& outputBinding = outputBindings[i];
        const TContainer& outputData = outputDataContainers[i].get();

        armnn::Tensor outputTensor(outputBinding.second, const_cast<float*>(outputData.data()));
        outputTensors.push_back(std::make_pair(outputBinding.first, outputTensor));
    }

    return outputTensors;
}

#define S_BOOL(name) enum class name {False=0, True=1};

S_BOOL(ImportMemory)
S_BOOL(DumpToDot)
S_BOOL(ExpectFile)
S_BOOL(OptionalArg)

int LoadModel(const char* filename,
              ITfLiteParser& parser,
              IRuntime& runtime,
              NetworkId& networkId,
              const std::vector<BackendId>& backendPreferences,
              ImportMemory enableImport,
              DumpToDot dumpToDot)
{
    std::ifstream stream(filename, std::ios::in | std::ios::binary);
    if (!stream.is_open())
    {
        ARMNN_LOG(error) << "Could not open model: " << filename;
        return OPEN_FILE_ERROR;
    }

    std::vector<uint8_t> contents((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    stream.close();

    auto model = parser.CreateNetworkFromBinary(contents);
    contents.clear();
    ARMNN_LOG(debug) << "Model loaded ok: " << filename;

    // Optimize backbone model
    OptimizerOptions options;
    options.m_ImportEnabled = enableImport != ImportMemory::False;
    auto optimizedModel = Optimize(*model, backendPreferences, runtime.GetDeviceSpec(), options);
    if (!optimizedModel)
    {
        ARMNN_LOG(fatal) << "Could not optimize the model:" << filename;
        return OPTIMIZE_NETWORK_ERROR;
    }

    if (dumpToDot != DumpToDot::False)
    {
        std::stringstream ss;
        ss << filename << ".dot";
        std::ofstream dotStream(ss.str().c_str(), std::ofstream::out);
        optimizedModel->SerializeToDot(dotStream);
        dotStream.close();
    }
    // Load model into runtime
    {
        std::string errorMessage;

        armnn::MemorySource memSource = options.m_ImportEnabled ? armnn::MemorySource::Malloc
                                                                : armnn::MemorySource::Undefined;
        INetworkProperties modelProps(false, memSource, memSource);
        Status status = runtime.LoadNetwork(networkId, std::move(optimizedModel), errorMessage, modelProps);
        if (status != Status::Success)
        {
            ARMNN_LOG(fatal) << "Could not load " << filename << " model into runtime: " << errorMessage;
            return LOAD_NETWORK_ERROR;
        }
    }

    return 0;
}

std::vector<float> LoadImage(const char* filename)
{
    if (strlen(filename) == 0)
    {
        return std::vector<float>(1920*10180*3, 0.0f);
    }
    struct Memory
    {
        ~Memory() {stbi_image_free(m_Data);}
        bool IsLoaded() const { return m_Data != nullptr;}

        unsigned char* m_Data;
    };

    std::vector<float> image;

    int width;
    int height;
    int channels;

    Memory mem = {stbi_load(filename, &width, &height, &channels, 3)};
    if (!mem.IsLoaded())
    {
        ARMNN_LOG(error) << "Could not load input image file: " << filename;
        return image;
    }

    if (width != 1920 || height != 1080 || channels != 3)
    {
        ARMNN_LOG(error) << "Input image has wong dimension: " << width << "x" << height << "x" << channels << ". "
          " Expected 1920x1080x3.";
        return image;
    }

    image.resize(1920*1080*3);

    // Expand to float. Does this need de-gamma?
    for (unsigned int idx=0; idx <= 1920*1080*3; idx++)
    {
        image[idx] = static_cast<float>(mem.m_Data[idx]) /255.0f;
    }

    return image;
}


bool ValidateFilePath(std::string& file, ExpectFile expectFile)
{
    if (!ghc::filesystem::exists(file))
    {
        std::cerr << "Given file path " << file << " does not exist" << std::endl;
        return false;
    }
    if (!ghc::filesystem::is_regular_file(file) && expectFile == ExpectFile::True)
    {
        std::cerr << "Given file path " << file << " is not a regular file" << std::endl;
        return false;
    }
    return true;
}

void CheckAccuracy(std::vector<float>* toDetector0, std::vector<float>* toDetector1,
                   std::vector<float>* toDetector2, std::vector<float>* detectorOutput,
                   const std::vector<yolov3::Detection>& nmsOut, const std::vector<std::string>& filePaths)
{
    std::ifstream pathStream;
    std::vector<float> expected;
    std::vector<std::vector<float>*> outputs;
    float compare = 0;
    unsigned int count = 0;

    //Push back output vectors from inference for use in loop
    outputs.push_back(toDetector0);
    outputs.push_back(toDetector1);
    outputs.push_back(toDetector2);
    outputs.push_back(detectorOutput);

    for (unsigned int i = 0; i < outputs.size(); ++i)
    {
        // Reading expected output files and assigning them to @expected. Close and Clear to reuse stream and clean RAM
        pathStream.open(filePaths[i]);
        if (!pathStream.is_open())
        {
            ARMNN_LOG(error) << "Expected output file can not be opened: " << filePaths[i];
            continue;
        }

        expected.assign(std::istream_iterator<float>(pathStream), {});
        pathStream.close();
        pathStream.clear();

        // Ensure each vector is the same length
        if (expected.size() != outputs[i]->size())
        {
            ARMNN_LOG(error) << "Expected output size does not match actual output size: " << filePaths[i];
        }
        else
        {
            count = 0;

            // Compare abs(difference) with tolerance to check for value by value equality
            for (unsigned int j = 0; j < outputs[i]->size(); ++j)
            {
                compare = std::abs(expected[j] - outputs[i]->at(j));
                if (compare > 0.001f)
                {
                    count++;
                }
            }
            if (count > 0)
            {
                ARMNN_LOG(error) << count << " output(s) do not match expected values in: " << filePaths[i];
            }
        }
    }

    pathStream.open(filePaths[4]);
    if (!pathStream.is_open())
    {
        ARMNN_LOG(error) << "Expected output file can not be opened: " << filePaths[4];
    }
    else
    {
        expected.assign(std::istream_iterator<float>(pathStream), {});
        pathStream.close();
        pathStream.clear();
        unsigned int y = 0;
        unsigned int numOfMember = 6;
        std::vector<float> intermediate;

        for (auto& detection: nmsOut)
        {
            for (unsigned int x = y * numOfMember; x < ((y * numOfMember) + numOfMember); ++x)
            {
                intermediate.push_back(expected[x]);
            }
            if (!yolov3::compare_detection(detection, intermediate))
            {
                ARMNN_LOG(error) << "Expected NMS output does not match: Detection " << y + 1;
            }
            intermediate.clear();
            y++;
        }
    }
}

struct ParseArgs
{
    ParseArgs(int ac, char *av[]) : options{"TfLiteYoloV3Big-Armnn",
                                            "Executes YoloV3Big using ArmNN. YoloV3Big consists "
                                            "of 3 parts: A backbone TfLite model, a detector TfLite "
                                            "model, and None Maximum Suppression. All parts are "
                                            "executed successively."}
    {
        options.add_options()
                ("b,backbone-path",
                 "File path where the TfLite model for the yoloV3big backbone "
                 "can be found e.g. mydir/yoloV3big_backbone.tflite",
                 cxxopts::value<std::string>())

               ("c,comparison-files",
                "Defines the expected outputs for the model "
                "of yoloV3big e.g. 'mydir/file1.txt,mydir/file2.txt,mydir/file3.txt,mydir/file4.txt'->InputToDetector1"
                " will be tried first then InputToDetector2 then InputToDetector3 then the Detector Output and finally"
                " the NMS output. NOTE: Files are passed as comma separated list without whitespaces.",
                cxxopts::value<std::vector<std::string>>()->default_value({}))

                ("d,detector-path",
                 "File path where the TfLite model for the yoloV3big "
                 "detector can be found e.g.'mydir/yoloV3big_detector.tflite'",
                 cxxopts::value<std::string>())

                ("h,help", "Produce help message")

                ("i,image-path",
                 "File path to a 1080x1920 jpg image that should be "
                 "processed e.g. 'mydir/example_img_180_1920.jpg'",
                 cxxopts::value<std::string>())

                ("B,preferred-backends-backbone",
                 "Defines the preferred backends to run the backbone model "
                 "of yoloV3big e.g. 'GpuAcc,CpuRef' -> GpuAcc will be tried "
                 "first before falling back to CpuRef. NOTE: Backends are passed "
                 "as comma separated list without whitespaces.",
                 cxxopts::value<std::vector<std::string>>()->default_value("GpuAcc,CpuRef"))

                ("D,preferred-backends-detector",
                 "Defines the preferred backends to run the detector model "
                 "of yoloV3big e.g. 'CpuAcc,CpuRef' -> CpuAcc will be tried "
                 "first before falling back to CpuRef. NOTE: Backends are passed "
                 "as comma separated list without whitespaces.",
                 cxxopts::value<std::vector<std::string>>()->default_value("CpuAcc,CpuRef"))

                ("M, model-to-dot",
                 "Dump the optimized model to a dot file for debugging/analysis",
                 cxxopts::value<bool>()->default_value("false"))

                ("Y, dynamic-backends-path",
                 "Define a path from which to load any dynamic backends.",
                 cxxopts::value<std::string>());

        auto result = options.parse(ac, av);

        if (result.count("help"))
        {
            std::cout << options.help() << "\n";
            exit(EXIT_SUCCESS);
        }


        backboneDir = GetPathArgument(result, "backbone-path", ExpectFile::True, OptionalArg::False);

        comparisonFiles = GetPathArgument(result["comparison-files"].as<std::vector<std::string>>(), OptionalArg::True);

        detectorDir = GetPathArgument(result, "detector-path", ExpectFile::True, OptionalArg::False);

        imageDir    = GetPathArgument(result, "image-path", ExpectFile::True, OptionalArg::True);

        dynamicBackendPath = GetPathArgument(result, "dynamic-backends-path", ExpectFile::False, OptionalArg::True);

        prefBackendsBackbone = GetBackendIDs(result["preferred-backends-backbone"].as<std::vector<std::string>>());
        LogBackendsInfo(prefBackendsBackbone, "Backbone");
        prefBackendsDetector = GetBackendIDs(result["preferred-backends-detector"].as<std::vector<std::string>>());
        LogBackendsInfo(prefBackendsDetector, "detector");

        dumpToDot = result["model-to-dot"].as<bool>() ? DumpToDot::True : DumpToDot::False;
    }

    /// Takes a vector of backend strings and returns a vector of backendIDs
    std::vector<BackendId> GetBackendIDs(const std::vector<std::string>& backendStrings)
    {
        std::vector<BackendId> backendIDs;
        for (const auto& b : backendStrings)
        {
            backendIDs.push_back(BackendId(b));
        }
        return backendIDs;
    }

    /// Verifies if the program argument with the name argName contains a valid file path.
    /// Returns the valid file path string if given argument is associated a valid file path.
    /// Otherwise throws an exception.
    std::string GetPathArgument(cxxopts::ParseResult& result,
                                std::string&& argName,
                                ExpectFile expectFile,
                                OptionalArg isOptionalArg)
    {
        if (result.count(argName))
        {
            std::string path = result[argName].as<std::string>();
            if (!ValidateFilePath(path, expectFile))
            {
                std::stringstream ss;
                ss << "Argument given to" << argName << "is not a valid file path";
                throw cxxopts::option_syntax_exception(ss.str().c_str());
            }
            return path;
        }
        else
        {
            if (isOptionalArg == OptionalArg::True)
            {
                return "";
            }

            throw cxxopts::missing_argument_exception(argName);
        }
    }

    /// Assigns vector of strings to struct member variable
    std::vector<std::string> GetPathArgument(const std::vector<std::string>& pathStrings, OptionalArg isOptional)
    {
        if (pathStrings.size() < 5){
            if (isOptional == OptionalArg::True)
            {
                return std::vector<std::string>();
            }
            throw cxxopts::option_syntax_exception("Comparison files requires 5 file paths.");
        }

        std::vector<std::string> filePaths;
        for (auto& path : pathStrings)
        {
            filePaths.push_back(path);
            if (!ValidateFilePath(filePaths.back(), ExpectFile::True))
            {
                throw cxxopts::option_syntax_exception("Argument given to Comparison Files is not a valid file path");
            }
        }
        return filePaths;
    }

    /// Log info about assigned backends
    void LogBackendsInfo(std::vector<BackendId>& backends, std::string&& modelName)
    {
        std::string info;
        info = "Preferred backends for " + modelName + " set to [ ";
        for (auto const &backend : backends)
        {
            info = info + std::string(backend) + " ";
        }
        ARMNN_LOG(info) << info << "]";
    }

    // Member variables
    std::string backboneDir;
    std::vector<std::string> comparisonFiles;
    std::string detectorDir;
    std::string imageDir;
    std::string dynamicBackendPath;

    std::vector<BackendId> prefBackendsBackbone;
    std::vector<BackendId> prefBackendsDetector;

    cxxopts::Options options;

    DumpToDot dumpToDot;
};

int main(int argc, char* argv[])
{
    // Configure logging
    SetAllLoggingSinks(true, true, true);
    SetLogFilter(LogSeverity::Trace);

    // Check and get given program arguments
    ParseArgs progArgs = ParseArgs(argc, argv);

    // Create runtime
    IRuntime::CreationOptions runtimeOptions; // default

    if (!progArgs.dynamicBackendPath.empty())
    {
        std::cout << "Loading backends from" << progArgs.dynamicBackendPath << "\n";
        runtimeOptions.m_DynamicBackendsPath = progArgs.dynamicBackendPath;
    }

    auto runtime = IRuntime::Create(runtimeOptions);
    if (!runtime)
    {
        ARMNN_LOG(fatal) << "Could not create runtime.";
        return -1;
    }

    // Create TfLite Parsers
    ITfLiteParser::TfLiteParserOptions parserOptions;
    auto parser = ITfLiteParser::Create(parserOptions);

    // Load backbone model
    ARMNN_LOG(info) << "Loading backbone...";
    NetworkId backboneId;
    const DumpToDot dumpToDot = progArgs.dumpToDot;
    CHECK_OK(LoadModel(progArgs.backboneDir.c_str(),
                       *parser,
                       *runtime,
                       backboneId,
                       progArgs.prefBackendsBackbone,
                       ImportMemory::False,
                       dumpToDot));
    auto inputId = parser->GetNetworkInputBindingInfo(0, "inputs");
    auto bbOut0Id = parser->GetNetworkOutputBindingInfo(0, "input_to_detector_1");
    auto bbOut1Id = parser->GetNetworkOutputBindingInfo(0, "input_to_detector_2");
    auto bbOut2Id = parser->GetNetworkOutputBindingInfo(0, "input_to_detector_3");
    auto backboneProfile = runtime->GetProfiler(backboneId);
    backboneProfile->EnableProfiling(true);


    // Load detector model
    ARMNN_LOG(info) << "Loading detector...";
    NetworkId detectorId;
    CHECK_OK(LoadModel(progArgs.detectorDir.c_str(),
                       *parser,
                       *runtime,
                       detectorId,
                       progArgs.prefBackendsDetector,
                       ImportMemory::True,
                       dumpToDot));
    auto detectIn0Id = parser->GetNetworkInputBindingInfo(0, "input_to_detector_1");
    auto detectIn1Id = parser->GetNetworkInputBindingInfo(0, "input_to_detector_2");
    auto detectIn2Id = parser->GetNetworkInputBindingInfo(0, "input_to_detector_3");
    auto outputBoxesId = parser->GetNetworkOutputBindingInfo(0, "output_boxes");
    auto detectorProfile = runtime->GetProfiler(detectorId);

    // Load input from file
    ARMNN_LOG(info) << "Loading test image...";
    auto image = LoadImage(progArgs.imageDir.c_str());
    if (image.empty())
    {
        return LOAD_IMAGE_ERROR;
    }

    // Allocate the intermediate tensors
    std::vector<float> intermediateMem0(bbOut0Id.second.GetNumElements());
    std::vector<float> intermediateMem1(bbOut1Id.second.GetNumElements());
    std::vector<float> intermediateMem2(bbOut2Id.second.GetNumElements());
    std::vector<float> intermediateMem3(outputBoxesId.second.GetNumElements());

    // Setup inputs and outputs
    using BindingInfos = std::vector<armnn::BindingPointInfo>;
    using FloatTensors = std::vector<std::reference_wrapper<std::vector<float>>>;

    InputTensors bbInputTensors = MakeInputTensors(BindingInfos{ inputId },
                                                   FloatTensors{ image });
    OutputTensors bbOutputTensors = MakeOutputTensors(BindingInfos{ bbOut0Id, bbOut1Id, bbOut2Id },
                                                      FloatTensors{ intermediateMem0,
                                                                    intermediateMem1,
                                                                    intermediateMem2 });
    InputTensors detectInputTensors = MakeInputTensors(BindingInfos{ detectIn0Id,
                                                                     detectIn1Id,
                                                                     detectIn2Id } ,
                                                       FloatTensors{ intermediateMem0,
                                                                     intermediateMem1,
                                                                     intermediateMem2 });
    OutputTensors detectOutputTensors = MakeOutputTensors(BindingInfos{ outputBoxesId },
                                                          FloatTensors{ intermediateMem3 });

    static const int numIterations=2;
    using DurationUS = std::chrono::duration<double, std::micro>;
    std::vector<DurationUS> nmsDurations(0);
    std::vector<yolov3::Detection> filtered_boxes;
    nmsDurations.reserve(numIterations);
    for (int i=0; i < numIterations; i++)
    {
        // Execute backbone
        ARMNN_LOG(info) << "Running backbone...";
        runtime->EnqueueWorkload(backboneId, bbInputTensors, bbOutputTensors);

        // Execute detector
        ARMNN_LOG(info) << "Running detector...";
        runtime->EnqueueWorkload(detectorId, detectInputTensors, detectOutputTensors);

        // Execute NMS
        ARMNN_LOG(info) << "Running nms...";
        using clock = std::chrono::steady_clock;
        auto nmsStartTime = clock::now();
        yolov3::NMSConfig config;
        config.num_boxes = 127800;
        config.num_classes = 80;
        config.confidence_threshold = 0.9f;
        config.iou_threshold = 0.5f;
        filtered_boxes = yolov3::nms(config, intermediateMem3);
        auto nmsEndTime = clock::now();

        // Enable the profiling after the warm-up run
        if (i>0)
        {
            print_detection(std::cout, filtered_boxes);

            const auto nmsDuration = DurationUS(nmsStartTime - nmsEndTime);
            nmsDurations.push_back(nmsDuration);
        }
        backboneProfile->EnableProfiling(true);
        detectorProfile->EnableProfiling(true);
    }
    // Log timings to file
    std::ofstream backboneProfileStream("backbone.json");
    backboneProfile->Print(backboneProfileStream);
    backboneProfileStream.close();

    std::ofstream detectorProfileStream("detector.json");
    detectorProfile->Print(detectorProfileStream);
    detectorProfileStream.close();

    // Manually construct the json output
    std::ofstream nmsProfileStream("nms.json");
    nmsProfileStream << "{" << "\n";
    nmsProfileStream << R"(  "NmsTimings": {)" << "\n";
    nmsProfileStream << R"(    "raw": [)" << "\n";
    bool isFirst = true;
    for (auto duration : nmsDurations)
    {
        if (!isFirst)
        {
            nmsProfileStream << ",\n";
        }

        nmsProfileStream << "      " << duration.count();
        isFirst = false;
    }
    nmsProfileStream << "\n";
    nmsProfileStream << R"(    "units": "us")" << "\n";
    nmsProfileStream << "    ]" << "\n";
    nmsProfileStream << "  }" << "\n";
    nmsProfileStream << "}" << "\n";
    nmsProfileStream.close();

    if (progArgs.comparisonFiles.size() > 0)
    {
        CheckAccuracy(&intermediateMem0,
                      &intermediateMem1,
                      &intermediateMem2,
                      &intermediateMem3,
                      filtered_boxes,
                      progArgs.comparisonFiles);
    }

    ARMNN_LOG(info) << "Run completed";
    return 0;
}
