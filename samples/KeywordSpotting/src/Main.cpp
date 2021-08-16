//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>
#include "KeywordSpottingPipeline.hpp"
#include "CmdArgsParser.hpp"
#include "ArmnnNetworkExecutor.hpp"
#include "AudioCapture.hpp"

const std::string AUDIO_FILE_PATH = "--audio-file-path";
const std::string MODEL_FILE_PATH = "--model-file-path";
const std::string LABEL_PATH = "--label-path";
const std::string PREFERRED_BACKENDS = "--preferred-backends";
const std::string HELP = "--help";

/*
 * The accepted options for this Speech Recognition executable
 */
static std::map<std::string, std::string> CMD_OPTIONS = 
{
        {AUDIO_FILE_PATH,    "[REQUIRED] Path to the Audio file to run speech recognition on"},
        {MODEL_FILE_PATH,    "[REQUIRED] Path to the Speech Recognition model to use"},
        {PREFERRED_BACKENDS, "[OPTIONAL] Takes the preferred backends in preference order, separated by comma."
                             " For example: CpuAcc,GpuAcc,CpuRef. Accepted options: [CpuAcc, CpuRef, GpuAcc]."
                             " Defaults to CpuAcc,CpuRef"}
};

/*
 * Reads the user supplied backend preference, splits it by comma, and returns an ordered vector
 */
std::vector<armnn::BackendId> GetPreferredBackendList(const std::string& preferredBackends) 
{
    std::vector<armnn::BackendId> backends;
    std::stringstream ss(preferredBackends);

    while (ss.good()) 
    {
        std::string backend;
        std::getline(ss, backend, ',');
        backends.emplace_back(backend);
    }
    return backends;
}

//Labels for this model
std::map<int, std::string> labels = 
{
        {0,  "silence"},
        {1,  "unknown"},
        {2,  "yes"},
        {3,  "no"},
        {4,  "up"},
        {5,  "down"},
        {6,  "left"},
        {7,  "right"},
        {8,  "on"},
        {9,  "off"},
        {10, "stop"},
        {11, "go"}
};


int main(int argc, char* argv[]) 
{
    printf("ArmNN major version: %d\n", ARMNN_MAJOR_VERSION);
    std::map<std::string, std::string> options;

    //Read command line args
    int result = ParseOptions(options, CMD_OPTIONS, argv, argc);
    if (result != 0) 
    {
        return result;
    }

    // Create the ArmNN inference runner
    common::PipelineOptions pipelineOptions;
    pipelineOptions.m_ModelName = "DS_CNN_CLUSTERED_INT8";
    pipelineOptions.m_ModelFilePath = GetSpecifiedOption(options, MODEL_FILE_PATH);
    if (CheckOptionSpecified(options, PREFERRED_BACKENDS)) 
    {
        pipelineOptions.m_backends = GetPreferredBackendList(
            (GetSpecifiedOption(options, PREFERRED_BACKENDS)));
    } 
    else 
    {
        pipelineOptions.m_backends = {"CpuAcc", "CpuRef"};
    }

    kws::IPipelinePtr kwsPipeline = kws::CreatePipeline(pipelineOptions);

    //Extract audio data from sound file
    auto filePath = GetSpecifiedOption(options, AUDIO_FILE_PATH);
    std::vector<float> audioData = audio::AudioCapture::LoadAudioFile(filePath);

    audio::AudioCapture capture;
    //todo: read samples and stride from pipeline
    capture.InitSlidingWindow(audioData.data(), 
                              audioData.size(), 
                              kwsPipeline->getInputSamplesSize(), 
                              kwsPipeline->getInputSamplesSize()/2);

    //Loop through audio data buffer
    while (capture.HasNext()) 
    {
        std::vector<float> audioBlock = capture.Next();
        common::InferenceResults<int8_t> results;

        //Prepare input tensors
        std::vector<int8_t> preprocessedData = kwsPipeline->PreProcessing(audioBlock);
        //Run inference
        kwsPipeline->Inference(preprocessedData, results);
        //Decode output
        kwsPipeline->PostProcessing(results, labels,
                                    [](int index, std::string& label, float prob) -> void {
                                        printf("Keyword \"%s\", index %d:, probability %f\n",
                                               label.c_str(),
                                               index,
                                               prob);
                                    });
    }

    return 0;
}