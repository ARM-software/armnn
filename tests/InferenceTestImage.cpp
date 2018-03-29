//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "InferenceTestImage.hpp"

#include <boost/core/ignore_unused.hpp>
#include <boost/format.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <array>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace
{

unsigned int GetImageChannelIndex(ImageChannelLayout channelLayout, ImageChannel channel)
{
    switch (channelLayout)
    {
    case ImageChannelLayout::Rgb:
        return static_cast<unsigned int>(channel);
    case ImageChannelLayout::Bgr:
        return 2u - static_cast<unsigned int>(channel);
    default:
        throw UnknownImageChannelLayout(boost::str(boost::format("Unknown layout %1%")
            % static_cast<int>(channelLayout)));
    }
}

} // namespace

InferenceTestImage::InferenceTestImage(char const* filePath)
 : m_Width(0u)
 , m_Height(0u)
 , m_NumChannels(0u)
{
    int width;
    int height;
    int channels;

    using StbImageDataPtr = std::unique_ptr<unsigned char, decltype(&stbi_image_free)>;
    StbImageDataPtr stbData(stbi_load(filePath, &width, &height, &channels, 0), &stbi_image_free);

    if (stbData == nullptr)
    {
        throw InferenceTestImageLoadFailed(boost::str(boost::format("Could not load the image at %1%") % filePath));
    }

    if (width == 0 || height == 0)
    {
        throw InferenceTestImageLoadFailed(boost::str(boost::format("Could not load empty image at %1%") % filePath));
    }

    m_Width = boost::numeric_cast<unsigned int>(width);
    m_Height = boost::numeric_cast<unsigned int>(height);
    m_NumChannels = boost::numeric_cast<unsigned int>(channels);

    const unsigned int sizeInBytes = GetSizeInBytes();
    m_Data.resize(sizeInBytes);
    memcpy(m_Data.data(), stbData.get(), sizeInBytes);
}

std::tuple<uint8_t, uint8_t, uint8_t> InferenceTestImage::GetPixelAs3Channels(unsigned int x, unsigned int y) const
{
    if (x >= m_Width || y >= m_Height)
    {
        throw InferenceTestImageOutOfBoundsAccess(boost::str(boost::format("Attempted out of bounds image access. "
            "Requested (%1%, %2%). Maximum valid coordinates (%3%, %4%).") % x % y % (m_Width - 1) % (m_Height - 1)));
    }

    const unsigned int pixelOffset = x * GetNumChannels() + y * GetWidth() * GetNumChannels();
    const uint8_t* const pixelData = m_Data.data() + pixelOffset;
    BOOST_ASSERT(pixelData <= (m_Data.data() + GetSizeInBytes()));

    std::array<uint8_t, 3> outPixelData;
    outPixelData.fill(0);

    const unsigned int maxChannelsInPixel = std::min(GetNumChannels(), static_cast<unsigned int>(outPixelData.size()));
    for (unsigned int c = 0; c < maxChannelsInPixel; ++c)
    {
        outPixelData[c] = pixelData[c];
    }

    return std::make_tuple(outPixelData[0], outPixelData[1], outPixelData[2]);
}

void InferenceTestImage::Resize(unsigned int newWidth, unsigned int newHeight)
{
    if (newWidth == 0 || newHeight == 0)
    {
        throw InferenceTestImageResizeFailed(boost::str(boost::format("None of the dimensions passed to a resize "
            "operation can be zero. Requested width: %1%. Requested height: %2%.") % newWidth % newHeight));
    }

    if (newWidth == m_Width && newHeight == m_Height)
    {
        // nothing to do
        return;
    }

    std::vector<uint8_t> newData;
    newData.resize(newWidth * newHeight * GetNumChannels() * GetSingleElementSizeInBytes());

    // boost::numeric_cast<>() is used for user-provided data (protecting about overflows).
    // static_cast<> ok for internal data (assumes that, when internal data was originally provided by a user,
    // a boost::numeric_cast<>() handled the conversion).
    const int nW = boost::numeric_cast<int>(newWidth);
    const int nH = boost::numeric_cast<int>(newHeight);

    const int w = static_cast<int>(GetWidth());
    const int h = static_cast<int>(GetHeight());
    const int numChannels = static_cast<int>(GetNumChannels());

    const int res = stbir_resize_uint8(m_Data.data(), w, h, 0, newData.data(), nW, nH, 0, numChannels);
    if (res == 0)
    {
        throw InferenceTestImageResizeFailed("The resizing operation failed");
    }

    m_Data.swap(newData);
    m_Width = newWidth;
    m_Height = newHeight;
}

void InferenceTestImage::Write(WriteFormat format, const char* filePath) const
{
    const int w = static_cast<int>(GetWidth());
    const int h = static_cast<int>(GetHeight());
    const int numChannels = static_cast<int>(GetNumChannels());
    int res = 0;

    switch (format)
    {
    case WriteFormat::Png:
        {
            res = stbi_write_png(filePath, w, h, numChannels, m_Data.data(), 0);
            break;
        }
    case WriteFormat::Bmp:
        {
            res = stbi_write_bmp(filePath, w, h, numChannels, m_Data.data());
            break;
        }
    case WriteFormat::Tga:
        {
            res = stbi_write_tga(filePath, w, h, numChannels, m_Data.data());
            break;
        }
    default:
        throw InferenceTestImageWriteFailed(boost::str(boost::format("Unknown format %1%")
            % static_cast<int>(format)));
    }

    if (res == 0)
    {
        throw InferenceTestImageWriteFailed(boost::str(boost::format("An error occurred when writing to file %1%")
            % filePath));
    }
}

template <typename TProcessValueCallable>
std::vector<float> GetImageDataInArmNnLayoutAsFloats(ImageChannelLayout channelLayout,
    const InferenceTestImage& image,
    TProcessValueCallable processValue)
{
    const unsigned int h = image.GetHeight();
    const unsigned int w = image.GetWidth();

    std::vector<float> imageData;
    imageData.resize(h * w * 3);

    for (unsigned int j = 0; j < h; ++j)
    {
        for (unsigned int i = 0; i < w; ++i)
        {
            uint8_t r, g, b;
            std::tie(r, g, b) = image.GetPixelAs3Channels(i, j);

            // ArmNN order: C, H, W
            const unsigned int rDstIndex = GetImageChannelIndex(channelLayout, ImageChannel::R) * h * w + j * w + i;
            const unsigned int gDstIndex = GetImageChannelIndex(channelLayout, ImageChannel::G) * h * w + j * w + i;
            const unsigned int bDstIndex = GetImageChannelIndex(channelLayout, ImageChannel::B) * h * w + j * w + i;

            imageData[rDstIndex] = processValue(ImageChannel::R, float(r));
            imageData[gDstIndex] = processValue(ImageChannel::G, float(g));
            imageData[bDstIndex] = processValue(ImageChannel::B, float(b));
        }
    }

    return imageData;
}

std::vector<float> GetImageDataInArmNnLayoutAsNormalizedFloats(ImageChannelLayout layout,
    const InferenceTestImage& image)
{
    return GetImageDataInArmNnLayoutAsFloats(layout, image,
        [](ImageChannel channel, float value)
        {
            boost::ignore_unused(channel);
            return value / 255.f;
        });
}

std::vector<float> GetImageDataInArmNnLayoutAsFloatsSubtractingMean(ImageChannelLayout layout,
    const InferenceTestImage& image,
    const std::array<float, 3>& mean)
{
    return GetImageDataInArmNnLayoutAsFloats(layout, image,
        [layout, &mean](ImageChannel channel, float value)
        {
            const unsigned int channelIndex = GetImageChannelIndex(layout, channel);
            return value - mean[channelIndex];
        });
}

std::vector<float> GetImageDataAsNormalizedFloats(ImageChannelLayout layout,
                                                  const InferenceTestImage& image)
{
    std::vector<float> imageData;
    const unsigned int h = image.GetHeight();
    const unsigned int w = image.GetWidth();

    const unsigned int rDstIndex = GetImageChannelIndex(layout, ImageChannel::R);
    const unsigned int gDstIndex = GetImageChannelIndex(layout, ImageChannel::G);
    const unsigned int bDstIndex = GetImageChannelIndex(layout, ImageChannel::B);

    imageData.resize(h * w * 3);
    unsigned int offset = 0;

    for (unsigned int j = 0; j < h; ++j)
    {
        for (unsigned int i = 0; i < w; ++i)
        {
            uint8_t r, g, b;
            std::tie(r, g, b) = image.GetPixelAs3Channels(i, j);

            imageData[offset+rDstIndex] = float(r) / 255.0f;
            imageData[offset+gDstIndex] = float(g) / 255.0f;
            imageData[offset+bDstIndex] = float(b) / 255.0f;
            offset += 3;
        }
    }

    return imageData;
}