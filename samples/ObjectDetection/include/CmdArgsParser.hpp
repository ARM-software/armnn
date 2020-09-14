//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once
#include <string>
#include <map>
#include <iostream>

const std::string MODEL_NAME = "--model-name";
const std::string VIDEO_FILE_PATH = "--video-file-path";
const std::string MODEL_FILE_PATH = "--model-file-path";
const std::string OUTPUT_VIDEO_FILE_PATH = "--output-video-file-path";
const std::string LABEL_PATH = "--label-path";
const std::string PREFERRED_BACKENDS = "--preferred-backends";
const std::string HELP = "--help";

/*
 * The accepted options for this Object detection executable
 */
static std::map<std::string, std::string> CMD_OPTIONS = {
        {VIDEO_FILE_PATH, "[REQUIRED] Path to the video file to run object detection on"},
        {MODEL_FILE_PATH, "[REQUIRED] Path to the Object Detection model to use"},
        {LABEL_PATH, "[REQUIRED] Path to the label set for the provided model file. "
                     "Label file is should just be an ordered list, seperated by new line."},
        {MODEL_NAME, "[REQUIRED] The name of the model being used. Accepted options: YOLO_V3_TINY, SSD_MOBILE"},
        {OUTPUT_VIDEO_FILE_PATH, "[OPTIONAL] Path to the output video file with detections added in. "
                                 "If specified will save file to disk, else displays the output to screen"},
        {PREFERRED_BACKENDS, "[OPTIONAL] Takes the preferred backends in preference order, separated by comma."
                             " For example: CpuAcc,GpuAcc,CpuRef. Accepted options: [CpuAcc, CpuRef, GpuAcc]."
                             " Defaults to CpuAcc,CpuRef"}
};

/*
 * Checks that a particular option was specified by the user
 */
bool CheckOptionSpecified(const std::map<std::string, std::string>& options, const std::string& option);


/*
 * Retrieves the user provided option
 */
std::string GetSpecifiedOption(const std::map<std::string, std::string>& options, const std::string& option);


/*
 * Parses all the command line options provided by the user and stores in a map.
 */
int ParseOptions(std::map<std::string, std::string>& options, std::map<std::string, std::string>& acceptedOptions,
                 char *argv[], int argc);