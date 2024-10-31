//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Exceptions.hpp>
#include <VerificationHelpers.hpp>

#include <array>
#include <cstdint>
#include <vector>
#include <utility>

class InferenceTestImageException : public armnn::Exception
{
public:
    using Exception::Exception;
};

class InferenceTestImageLoadFailed : public InferenceTestImageException
{
public:
    using InferenceTestImageException::InferenceTestImageException;
};

class InferenceTestImageOutOfBoundsAccess : public InferenceTestImageException
{
public:
    using InferenceTestImageException::InferenceTestImageException;
};

class InferenceTestImageResizeFailed : public InferenceTestImageException
{
public:
    using InferenceTestImageException::InferenceTestImageException;
};

class InferenceTestImageWriteFailed : public InferenceTestImageException
{
public:
    using InferenceTestImageException::InferenceTestImageException;
};

class UnknownImageChannelLayout : public InferenceTestImageException
{
public:
    using InferenceTestImageException::InferenceTestImageException;
};

class InferenceTestImage
{
public:
    enum class WriteFormat
    {
        Png,
        Bmp,
        Tga
    };

    // Common names used to identify a channel in a pixel.
    enum class ResizingMethods
    {
        STB,
        BilinearAndNormalized,
    };

    explicit InferenceTestImage(const char* filePath);

    InferenceTestImage(InferenceTestImage&&) = delete;
    InferenceTestImage(const InferenceTestImage&) = delete;
    InferenceTestImage& operator=(const InferenceTestImage&) = delete;
    InferenceTestImage& operator=(InferenceTestImage&&) = delete;

    unsigned int GetWidth() const { return m_Width; }
    unsigned int GetHeight() const { return m_Height; }
    unsigned int GetNumChannels() const { return m_NumChannels; }
    unsigned int GetNumElements() const { return GetWidth() * GetHeight() * GetNumChannels(); }
    unsigned int GetSizeInBytes() const { return GetNumElements() * GetSingleElementSizeInBytes(); }

    // Returns the pixel identified by the given coordinates as a 3-channel value.
    // Channels beyond the third are dropped. If the image provides less than 3 channels, the non-existent
    // channels of the pixel will be filled with 0. Channels are returned in RGB order (that is, the first element
    // of the tuple corresponds to the Red channel, whereas the last element is the Blue channel).
    std::tuple<uint8_t, uint8_t, uint8_t> GetPixelAs3Channels(unsigned int x, unsigned int y) const;

    void StbResize(InferenceTestImage& im, const unsigned int newWidth, const unsigned int newHeight);


    std::vector<float> Resize(unsigned int newWidth,
                              unsigned int newHeight,
                              const armnn::CheckLocation& location,
                              const ResizingMethods meth = ResizingMethods::STB,
                              const std::array<float, 3>& mean = {{0.0, 0.0, 0.0}},
                              const std::array<float, 3>& stddev = {{1.0, 1.0, 1.0}},
                              const float scale = 255.0f);

    void Write(WriteFormat format, const char* filePath) const;

private:
    static unsigned int GetSingleElementSizeInBytes()
    {
        return sizeof(decltype(std::declval<InferenceTestImage>().m_Data[0]));
    }

    std::vector<uint8_t> m_Data;
    unsigned int m_Width;
    unsigned int m_Height;
    unsigned int m_NumChannels;
};

// Common names used to identify a channel in a pixel.
enum class ImageChannel
{
    R,
    G,
    B
};

// Channel layouts handled by the test framework.
enum class ImageChannelLayout
{
    Rgb,
    Bgr
};

// Reads the contents of an inference test image as 3-channel pixels whose channel values have been normalized (scaled)
// and now lie in the range [0,1]. Channel data is stored according to the ArmNN layout (CHW). The order in which
// channels appear in the resulting vector is defined by the provided layout.
std::vector<float> GetImageDataInArmNnLayoutAsNormalizedFloats(ImageChannelLayout layout,
                                                               const InferenceTestImage& image);

// Reads the contents of an inference test image as 3-channel pixels, whose value is the result of subtracting the mean
// from the values in the original image. Channel data is stored according to the ArmNN layout (CHW). The order in
// which channels appear in the resulting vector is defined by the provided layout. The order of the channels of the
// provided mean should also match the given layout.
std::vector<float> GetImageDataInArmNnLayoutAsFloatsSubtractingMean(ImageChannelLayout layout,
                                                                    const InferenceTestImage& image,
                                                                    const std::array<float, 3>& mean);

// Reads the contents of an inference test image as 3-channel pixels and returns the image data as normalized float
// values. The returned image stay in the original order (HWC) order. The C order may be changed according to the
// supplied layout value.
std::vector<float> GetImageDataAsNormalizedFloats(ImageChannelLayout layout,
                                                  const InferenceTestImage& image);
