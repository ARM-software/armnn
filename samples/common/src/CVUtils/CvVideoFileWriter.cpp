//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CvVideoFileWriter.hpp"

namespace common
{

void CvVideoFileWriter::Init(const std::string& outputVideo, int encoding, double fps, int width, int height)
{
    m_ready = m_cvWriter.open(outputVideo, cv::CAP_FFMPEG,
                              encoding,
                              fps,
                              cv::Size(width, height), true);
}


void CvVideoFileWriter::WriteFrame(std::shared_ptr<cv::Mat>& frame)
{
    if(m_cvWriter.isOpened())
    {
        cv::cvtColor(*frame, *frame, cv::COLOR_RGB2BGR);
        m_cvWriter.write(*frame);
    }
}

bool CvVideoFileWriter::IsReady() const
{
    return m_ready;
}

void CvVideoFileWriter::Close()
{
    m_cvWriter.release();
}
}// namespace common
