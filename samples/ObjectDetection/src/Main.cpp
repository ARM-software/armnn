//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CvVideoFrameReader.hpp"
#include "CvWindowOutput.hpp"
#include "CvVideoFileWriter.hpp"
#include "ObjectDetectionPipeline.hpp"
#include "CmdArgsParser.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <random>

const std::string MODEL_NAME = "--model-name";
const std::string VIDEO_FILE_PATH = "--video-file-path";
const std::string MODEL_FILE_PATH = "--model-file-path";
const std::string OUTPUT_VIDEO_FILE_PATH = "--output-video-file-path";
const std::string LABEL_PATH = "--label-path";
const std::string PREFERRED_BACKENDS = "--preferred-backends";
const std::string PROFILING_ENABLED = "--profiling_enabled";
const std::string HELP = "--help";

/*
 * The accepted options for this Object detection executable
 */
static std::map<std::string, std::string> CMD_OPTIONS = {
        {VIDEO_FILE_PATH, "[REQUIRED] Path to the video file to run object detection on"},
        {MODEL_FILE_PATH, "[REQUIRED] Path to the Object Detection model to use"},
        {LABEL_PATH, "[REQUIRED] Path to the label set for the provided model file. "
                     "Label file  should be an ordered list, separated by a new line."},
        {MODEL_NAME, "[REQUIRED] The name of the model being used. Accepted options: YOLO_V3_TINY, SSD_MOBILE"},
        {OUTPUT_VIDEO_FILE_PATH, "[OPTIONAL] Path to the output video file with detections added in. "
                                 "If specified will save file to disk, else displays the output to screen"},
        {PREFERRED_BACKENDS, "[OPTIONAL] Takes the preferred backends in preference order, separated by comma."
                             " For example: CpuAcc,GpuAcc,CpuRef. Accepted options: [CpuAcc, CpuRef, GpuAcc]."
                             " Defaults to CpuAcc,CpuRef"},
        {PROFILING_ENABLED, "[OPTIONAL] Enabling this option will print important ML related milestones timing"
                            "information in micro-seconds. By default, this option is disabled."
                            "Accepted options are true/false."}
};

/*
 * Reads the user supplied backend preference, splits it by comma, and returns an ordered vector
 */
std::vector<armnn::BackendId> GetPreferredBackendList(const std::string& preferredBackends)
{
    std::vector<armnn::BackendId> backends;
    std::stringstream ss(preferredBackends);

    while(ss.good())
    {
        std::string backend;
        std::getline( ss, backend, ',' );
        backends.emplace_back(backend);
    }
    return backends;
}

/*
 * Assigns a color to each label in the label set
 */
std::vector<std::tuple<std::string, common::BBoxColor>> AssignColourToLabel(const std::string& pathToLabelFile)
{
    std::ifstream in(pathToLabelFile);
    std::vector<std::tuple<std::string, common::BBoxColor>> labels;

    std::string str;
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,255);

    while (std::getline(in, str))
    {
        if(!str.empty())
        {
            common::BBoxColor c{
                .colorCode = std::make_tuple(distribution(generator),
                                             distribution(generator),
                                             distribution(generator))
            };
            auto bboxInfo = std::make_tuple (str, c);

            labels.emplace_back(bboxInfo);
        }
    }
    return labels;
}

std::tuple<std::unique_ptr<common::IFrameReader<cv::Mat>>,
           std::unique_ptr<common::IFrameOutput<cv::Mat>>>
           GetFrameSourceAndSink(const std::map<std::string, std::string>& options) {

    std::unique_ptr<common::IFrameReader<cv::Mat>> readerPtr;

    std::unique_ptr<common::CvVideoFrameReader> reader = std::make_unique<common::CvVideoFrameReader>();
    reader->Init(GetSpecifiedOption(options, VIDEO_FILE_PATH));

    auto enc = reader->GetSourceEncodingInt();
    auto fps = reader->GetSourceFps();
    auto w = reader->GetSourceWidth();
    auto h = reader->GetSourceHeight();
    if (!reader->ConvertToRGB())
    {
        readerPtr = std::move(std::make_unique<common::CvVideoFrameReaderRgbWrapper>(std::move(reader)));
    }
    else
    {
        readerPtr = std::move(reader);
    }

    if(CheckOptionSpecified(options, OUTPUT_VIDEO_FILE_PATH))
    {
        std::string outputVideo = GetSpecifiedOption(options, OUTPUT_VIDEO_FILE_PATH);
        auto writer = std::make_unique<common::CvVideoFileWriter>();
        writer->Init(outputVideo, enc, fps, w, h);

        return std::make_tuple<>(std::move(readerPtr), std::move(writer));
    }
    else
    {
        auto writer = std::make_unique<common::CvWindowOutput>();
        writer->Init("Processed Video");
        return std::make_tuple<>(std::move(readerPtr), std::move(writer));
    }
}

int main(int argc, char *argv[])
{
    std::map<std::string, std::string> options;

    int result = ParseOptions(options, CMD_OPTIONS, argv, argc);
    if (result != 0)
    {
        return result;
    }

    // Create the network options
    common::PipelineOptions pipelineOptions;
    pipelineOptions.m_ModelFilePath = GetSpecifiedOption(options, MODEL_FILE_PATH);
    pipelineOptions.m_ModelName = GetSpecifiedOption(options, MODEL_NAME);

    if (CheckOptionSpecified(options, PROFILING_ENABLED))
    {
        pipelineOptions.m_ProfilingEnabled = GetSpecifiedOption(options, PROFILING_ENABLED) == "true";
    }
    if(CheckOptionSpecified(options, PREFERRED_BACKENDS))
    {
        pipelineOptions.m_backends = GetPreferredBackendList((GetSpecifiedOption(options, PREFERRED_BACKENDS)));
    }
    else
    {
        pipelineOptions.m_backends = {"CpuAcc", "CpuRef"};
    }

    auto labels = AssignColourToLabel(GetSpecifiedOption(options, LABEL_PATH));

    common::Profiling profiling(pipelineOptions.m_ProfilingEnabled);
    profiling.ProfilingStart();
    od::IPipelinePtr objectDetectionPipeline = od::CreatePipeline(pipelineOptions);

    auto inputAndOutput = GetFrameSourceAndSink(options);
    std::unique_ptr<common::IFrameReader<cv::Mat>> reader = std::move(std::get<0>(inputAndOutput));
    std::unique_ptr<common::IFrameOutput<cv::Mat>> sink = std::move(std::get<1>(inputAndOutput));

    if (!sink->IsReady())
    {
        std::cerr << "Failed to open video writer.";
        return 1;
    }

    common::InferenceResults<float> results;

    std::shared_ptr<cv::Mat> frame = reader->ReadFrame();

    //pre-allocate frames
    cv::Mat processed;

    while(!reader->IsExhausted(frame))
    {
        objectDetectionPipeline->PreProcessing(*frame, processed);
        objectDetectionPipeline->Inference(processed, results);
        objectDetectionPipeline->PostProcessing(results,
                                                [&frame, &labels](od::DetectedObjects detects) -> void {
            AddInferenceOutputToFrame(detects, *frame, labels);
        });

        sink->WriteFrame(frame);
        frame = reader->ReadFrame();
    }
    sink->Close();
    profiling.ProfilingStopAndPrintUs("Overall compute time");
    return 0;
}
