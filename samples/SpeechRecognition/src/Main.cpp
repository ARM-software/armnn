//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>

#include "CmdArgsParser.hpp"
#include "ArmnnNetworkExecutor.hpp"
#include "AudioCapture.hpp"
#include "SpeechRecognitionPipeline.hpp"
#include "Wav2LetterMFCC.hpp"

using InferenceResult = std::vector<int8_t>;
using InferenceResults = std::vector<InferenceResult>;

const std::string AUDIO_FILE_PATH = "--audio-file-path";
const std::string MODEL_FILE_PATH = "--model-file-path";
const std::string LABEL_PATH = "--label-path";
const std::string PREFERRED_BACKENDS = "--preferred-backends";
const std::string HELP = "--help";

std::map<int, std::string> labels = 
{
        {0,  "a"},
        {1,  "b"},
        {2,  "c"},
        {3,  "d"},
        {4,  "e"},
        {5,  "f"},
        {6,  "g"},
        {7,  "h"},
        {8,  "i"},
        {9,  "j"},
        {10, "k"},
        {11, "l"},
        {12, "m"},
        {13, "n"},
        {14, "o"},
        {15, "p"},
        {16, "q"},
        {17, "r"},
        {18, "s"},
        {19, "t"},
        {20, "u"},
        {21, "v"},
        {22, "w"},
        {23, "x"},
        {24, "y"},
        {25, "z"},
        {26, "\'"},
        {27, " "},
        {28, "$"}
};

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

int main(int argc, char* argv[]) 
{
    bool isFirstWindow = true;
    std::string currentRContext = "";

    std::map<std::string, std::string> options;

    int result = ParseOptions(options, CMD_OPTIONS, argv, argc);
    if (result != 0) 
    {
        return result;
    }

    // Create the network options
    common::PipelineOptions pipelineOptions;
    pipelineOptions.m_ModelFilePath = GetSpecifiedOption(options, MODEL_FILE_PATH);
    pipelineOptions.m_ModelName = "Wav2Letter";
    if (CheckOptionSpecified(options, PREFERRED_BACKENDS)) 
    {
        pipelineOptions.m_backends = GetPreferredBackendList((GetSpecifiedOption(options, PREFERRED_BACKENDS)));
    } 
    else 
    {
        pipelineOptions.m_backends = {"CpuAcc", "CpuRef"};
    }

    asr::IPipelinePtr asrPipeline = asr::CreatePipeline(pipelineOptions, labels);

    audio::AudioCapture capture;
    std::vector<float> audioData = audio::AudioCapture::LoadAudioFile(GetSpecifiedOption(options, AUDIO_FILE_PATH));
    capture.InitSlidingWindow(audioData.data(), audioData.size(), asrPipeline->getInputSamplesSize(),
                              asrPipeline->getSlidingWindowOffset());

    while (capture.HasNext()) 
    {
        std::vector<float> audioBlock = capture.Next();
        InferenceResults results;

        std::vector<int8_t> preprocessedData = asrPipeline->PreProcessing(audioBlock);
        asrPipeline->Inference<int8_t>(preprocessedData, results);
        asrPipeline->PostProcessing<int8_t>(results, isFirstWindow, !capture.HasNext(), currentRContext);
    }

    return 0;
}