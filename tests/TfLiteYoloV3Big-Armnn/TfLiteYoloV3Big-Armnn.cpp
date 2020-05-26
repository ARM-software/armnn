//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
//#include "../InferenceTest.hpp"
//#include "../ImagePreprocessor.hpp"
#include "armnnTfLiteParser/ITfLiteParser.hpp"

#include "NMS.hpp"

#include <stb/stb_image.h>

#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Logging.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <chrono>
#include <iostream>
#include <fstream>

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
                                            const std::vector<TContainer>& inputDataContainers)
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
        const TContainer& inputData = inputDataContainers[i];

        armnn::ConstTensor inputTensor(inputBinding.second, inputData.data());
        inputTensors.push_back(std::make_pair(inputBinding.first, inputTensor));
    }

    return inputTensors;
}

template<typename TContainer>
inline armnn::OutputTensors MakeOutputTensors(const std::vector<armnn::BindingPointInfo>& outputBindings,
                                              const std::vector<TContainer>& outputDataContainers)
{
    armnn::OutputTensors outputTensors;

    const size_t numOutputs = outputBindings.size();
    if (numOutputs != outputDataContainers.size())
    {
        throw armnn::Exception("Mismatching vectors");
    }

    for (size_t i = 0; i < numOutputs; i++)
    {
        const armnn::BindingPointInfo& outputBinding = outputBindings[i];
        const TContainer& outputData = outputDataContainers[i];

        armnn::Tensor outputTensor(outputBinding.second, const_cast<float*>(outputData.data()));
        outputTensors.push_back(std::make_pair(outputBinding.first, outputTensor));
    }

    return outputTensors;
}

int LoadModel(const char* filename,
              ITfLiteParser& parser,
              IRuntime& runtime,
              NetworkId& networkId,
              const std::vector<BackendId>& backendPreferences)
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
    auto optimizedModel = Optimize(*model, backendPreferences, runtime.GetDeviceSpec());
    if (!optimizedModel)
    {
        ARMNN_LOG(fatal) << "Could not optimize the model:" << filename;
        return OPTIMIZE_NETWORK_ERROR;
    }

    // Load backbone model into runtime
    {
        std::string errorMessage;
        INetworkProperties modelProps;
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

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        ARMNN_LOG(error) << "Expected arguments: {PathToModels} {PathToData}";
    }
    std::string modelsPath(argv[1]);
    std::string imagePath(argv[2]);

    std::string backboneModelFile = modelsPath + "yolov3_1080_1920_backbone_int8.tflite";
    std::string detectorModelFile = modelsPath + "yolov3_1080_1920_detector_fp32.tflite";
    std::string imageFile = imagePath + "1080_1920.jpg";

    // Configure the logging
    SetAllLoggingSinks(true, true, true);
    SetLogFilter(LogSeverity::Trace);


    // Create runtime
    IRuntime::CreationOptions runtimeOptions; // default
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
    CHECK_OK(LoadModel(backboneModelFile.c_str(), *parser, *runtime, backboneId, {"GpuAcc", "CpuRef"}));
    auto inputId = parser->GetNetworkInputBindingInfo(0, "inputs");
    auto bbOut0Id = parser->GetNetworkOutputBindingInfo(0, "input_to_detector_1");
    auto bbOut1Id = parser->GetNetworkOutputBindingInfo(0, "input_to_detector_2");
    auto bbOut2Id = parser->GetNetworkOutputBindingInfo(0, "input_to_detector_3");
    auto backboneProfile = runtime->GetProfiler(backboneId);
    backboneProfile->EnableProfiling(true);

    // Load detector model
    ARMNN_LOG(info) << "Loading detector...";
    NetworkId detectorId;
    CHECK_OK(LoadModel(detectorModelFile.c_str(), *parser, *runtime, detectorId, {"CpuAcc", "CpuRef"}));
    auto detectIn0Id = parser->GetNetworkInputBindingInfo(0, "input_to_detector_1");
    auto detectIn1Id = parser->GetNetworkInputBindingInfo(0, "input_to_detector_2");
    auto detectIn2Id = parser->GetNetworkInputBindingInfo(0, "input_to_detector_3");
    auto outputBoxesId = parser->GetNetworkOutputBindingInfo(0, "output_boxes");
    auto detectorProfile = runtime->GetProfiler(detectorId);

    // Load input from file
    ARMNN_LOG(info) << "Loading test image...";
    auto image = LoadImage(imageFile.c_str());
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
    using FloatTensors = std::vector<std::vector<float>>;

    InputTensors bbInputTensors = MakeInputTensors(BindingInfos{inputId},
                                                   FloatTensors{std::move(image)});
    OutputTensors bbOutputTensors = MakeOutputTensors(BindingInfos{bbOut0Id, bbOut1Id, bbOut2Id},
                                                      FloatTensors{intermediateMem0,
                                                                   intermediateMem1,
                                                                   intermediateMem2});
    InputTensors detectInputTensors = MakeInputTensors(BindingInfos{detectIn0Id,
                                                                    detectIn1Id,
                                                                    detectIn2Id},
                                                       FloatTensors{intermediateMem0,
                                                                    intermediateMem1,
                                                                    intermediateMem2});
    OutputTensors detectOutputTensors = MakeOutputTensors(BindingInfos{outputBoxesId},
                                                          FloatTensors{intermediateMem3});

    static const int numIterations=2;
    using DurationUS = std::chrono::duration<double, std::micro>;
    std::vector<DurationUS> nmsDurations(0);
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
        auto filtered_boxes = yolov3::nms(config, intermediateMem3);
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

    ARMNN_LOG(info) << "Run completed";
    return 0;
}