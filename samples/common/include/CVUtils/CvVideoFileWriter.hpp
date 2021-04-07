//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IFrameOutput.hpp"
#include <opencv2/opencv.hpp>

namespace common
{

class CvVideoFileWriter : public IFrameOutput<cv::Mat> {
public:
    /**
     * @brief Default constructor.
     *
     * Underlying open cv video writer object will be instantiated.
     */
    CvVideoFileWriter() = default;

    ~CvVideoFileWriter() override = default;

    /**
     * @brief Initialises video file writer.
     *
     * Opens opencv writer with given params. FFMPEG backend is used.
     *
     * @param outputVideo path to the video file.
     * @param encoding cv::CAP_PROP_FOURCC code.
     * @param fps target frame rate.
     * @param width target frame width.
     * @param height target frame height.
     *
     */
    void Init(const std::string& outputVideo, int encoding, double fps, int width, int height);

    /**
     * Writes frame to the file using opencv writer.
     *
     * @param frame data to write.
     */
    void WriteFrame(std::shared_ptr<cv::Mat>& frame) override;

    /**
     * Releases opencv writer.
     */
    void Close() override;

    /**
     * Checks if opencv writer was successfully opened.
     * @return true is underlying writer is ready to be used, false otherwise.
     */
    bool IsReady() const override;

private:
    cv::VideoWriter m_cvWriter{};
    bool m_ready = false;
};
}// namespace common