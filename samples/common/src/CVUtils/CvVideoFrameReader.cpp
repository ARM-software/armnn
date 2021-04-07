//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//


#include "CvVideoFrameReader.hpp"

namespace common
{

std::shared_ptr<cv::Mat> CvVideoFrameReader::ReadFrame()
{
    // opencv copies data anyway
    cv::Mat captureFrame;
    m_capture.read(captureFrame);
    return std::make_shared<cv::Mat>(std::move(captureFrame));
}

bool CvVideoFrameReader::IsExhausted(const std::shared_ptr<cv::Mat>& frame) const
{
    assert(frame!=nullptr);
    return frame->empty();
}

void CvVideoFrameReader::CheckIsOpen(const std::string& source)
{
    if (!m_capture.isOpened())
    {
        throw std::runtime_error("Failed to open video capture for the source = " + source);
    }
}

void CvVideoFrameReader::Init(const std::string& source)
{
    m_capture.open(source);
    CheckIsOpen(source);
}

int CvVideoFrameReader::GetSourceWidth() const
{
    return static_cast<int>(lround(m_capture.get(cv::CAP_PROP_FRAME_WIDTH)));
}

int CvVideoFrameReader::GetSourceHeight() const
{
    return static_cast<int>(lround(m_capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
}

double CvVideoFrameReader::GetSourceFps() const
{
    return m_capture.get(cv::CAP_PROP_FPS);
}

bool CvVideoFrameReader::ConvertToRGB()
{
    m_capture.set(cv::CAP_PROP_CONVERT_RGB, 1.0);
    return static_cast<bool>(m_capture.get(cv::CAP_PROP_CONVERT_RGB));
}

std::string CvVideoFrameReader::GetSourceEncoding() const
{
    char fourccStr[5];
    auto fourcc = (int)m_capture.get(cv::CAP_PROP_FOURCC);
    sprintf(fourccStr,"%c%c%c%c",fourcc & 0xFF, (fourcc >> 8) & 0xFF, (fourcc >> 16) & 0xFF, (fourcc >> 24) & 0xFF);
    return fourccStr;
}

int CvVideoFrameReader::GetSourceEncodingInt() const
{
    return (int)m_capture.get(cv::CAP_PROP_FOURCC);
}

int CvVideoFrameReader::GetFrameCount() const
{
    return static_cast<int>(lround(m_capture.get(cv::CAP_PROP_FRAME_COUNT)));
};

std::shared_ptr<cv::Mat> CvVideoFrameReaderRgbWrapper::ReadFrame()
{
    auto framePtr = m_reader->ReadFrame();
    if (!IsExhausted(framePtr))
    {
        cv::cvtColor(*framePtr, *framePtr, cv::COLOR_BGR2RGB);
    }
    return framePtr;
}

bool CvVideoFrameReaderRgbWrapper::IsExhausted(const std::shared_ptr<cv::Mat>& frame) const
{
    return m_reader->IsExhausted(frame);
}

CvVideoFrameReaderRgbWrapper::CvVideoFrameReaderRgbWrapper(std::unique_ptr<common::CvVideoFrameReader> reader):
        m_reader(std::move(reader))
{}

}// namespace common