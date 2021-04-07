//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "IFrameOutput.hpp"
#include <opencv2/opencv.hpp>

namespace common
{

class CvWindowOutput : public IFrameOutput<cv::Mat> {
public:

    CvWindowOutput() = default;

    ~CvWindowOutput() override = default;

    /**
     * @brief Creates a named window.
     *
     * Uses opencv to create a window with given name.
     *
     * @param windowName opencv window name.
     *
     */
    void Init(const std::string& windowName);

    /**
     * Writes frame to the window.
     *
     * @param frame data to write.
     */
    void WriteFrame(std::shared_ptr<cv::Mat>& frame) override;

    /**
     * Releases all windows.
     */
    void Close() override;

    /**
     * Always true.
     * @return true.
     */
    bool IsReady() const override;

private:
    std::string m_windowName;

};
}// namespace common