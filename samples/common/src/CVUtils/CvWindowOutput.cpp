//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CvWindowOutput.hpp"

namespace common
{

void CvWindowOutput::Init(const std::string& windowName)
{
    m_windowName = windowName;
    cv::namedWindow(m_windowName, cv::WINDOW_AUTOSIZE);
}

void CvWindowOutput::WriteFrame(std::shared_ptr<cv::Mat>& frame)
{
    cv::cvtColor(*frame, *frame, cv::COLOR_RGB2BGR);
    cv::imshow( m_windowName, *frame);
    cv::waitKey(30);
}

void CvWindowOutput::Close()
{
    cv::destroyWindow(m_windowName);
}

bool CvWindowOutput::IsReady() const
{
    return true;
}
}// namespace common