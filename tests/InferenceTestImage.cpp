//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "InferenceTestImage.hpp"

#include <boost/core/ignore_unused.hpp>
#include <boost/format.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <array>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

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

inline float Lerp(float a, float b, float w)
{
    return w * b + (1.f - w) * a;
}

inline void PutData(std::vector<float> & data,
                    const unsigned int width,
                    const unsigned int x,
                    const unsigned int y,
                    const unsigned int c,
                    float value)
{
    data[(3*((y*width)+x)) + c] = value;
}

std::vector<float> ResizeBilinearAndNormalize(const InferenceTestImage & image,
                                              const unsigned int outputWidth,
                                              const unsigned int outputHeight,
                                              const std::array<float, 3>& mean,
                                              const std::array<float, 3>& stddev)
{
    std::vector<float> out;
    out.resize(outputWidth * outputHeight * 3);

    // We follow the definition of TensorFlow and AndroidNN: the top-left corner of a texel in the output
    // image is projected into the input image to figure out the interpolants and weights. Note that this
    // will yield different results than if projecting the centre of output texels.

    const unsigned int inputWidth = image.GetWidth();
    const unsigned int inputHeight = image.GetHeight();

    // How much to scale pixel coordinates in the output image to get the corresponding pixel coordinates
    // in the input image.
    const float scaleY = boost::numeric_cast<float>(inputHeight) / boost::numeric_cast<float>(outputHeight);
    const float scaleX = boost::numeric_cast<float>(inputWidth) / boost::numeric_cast<float>(outputWidth);

    uint8_t rgb_x0y0[3];
    uint8_t rgb_x1y0[3];
    uint8_t rgb_x0y1[3];
    uint8_t rgb_x1y1[3];

    for (unsigned int y = 0; y < outputHeight; ++y)
    {
        // Corresponding real-valued height coordinate in input image.
        const float iy = boost::numeric_cast<float>(y) * scaleY;

        // Discrete height coordinate of top-left texel (in the 2x2 texel area used for interpolation).
        const float fiy = floorf(iy);
        const unsigned int y0 = boost::numeric_cast<unsigned int>(fiy);

        // Interpolation weight (range [0,1])
        const float yw = iy - fiy;

        for (unsigned int x = 0; x < outputWidth; ++x)
        {
            // Real-valued and discrete width coordinates in input image.
            const float ix = boost::numeric_cast<float>(x) * scaleX;
            const float fix = floorf(ix);
            const unsigned int x0 = boost::numeric_cast<unsigned int>(fix);

            // Interpolation weight (range [0,1]).
            const float xw = ix - fix;

            // Discrete width/height coordinates of texels below and to the right of (x0, y0).
            const unsigned int x1 = std::min(x0 + 1, inputWidth - 1u);
            const unsigned int y1 = std::min(y0 + 1, inputHeight - 1u);

            std::tie(rgb_x0y0[0], rgb_x0y0[1], rgb_x0y0[2]) = image.GetPixelAs3Channels(x0, y0);
            std::tie(rgb_x1y0[0], rgb_x1y0[1], rgb_x1y0[2]) = image.GetPixelAs3Channels(x1, y0);
            std::tie(rgb_x0y1[0], rgb_x0y1[1], rgb_x0y1[2]) = image.GetPixelAs3Channels(x0, y1);
            std::tie(rgb_x1y1[0], rgb_x1y1[1], rgb_x1y1[2]) = image.GetPixelAs3Channels(x1, y1);

            for (unsigned c=0; c<3; ++c)
            {
                const float ly0 = Lerp(float(rgb_x0y0[c]), float(rgb_x1y0[c]), xw);
                const float ly1 = Lerp(float(rgb_x0y1[c]), float(rgb_x1y1[c]), xw);
                const float l = Lerp(ly0, ly1, yw);
                PutData(out, outputWidth, x, y, c, ((l/255.0f) - mean[c])/stddev[c]);
            }
        }
    }
    return out;
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


void InferenceTestImage::StbResize(InferenceTestImage& im, const unsigned int newWidth, const unsigned int newHeight)
{
    std::vector<uint8_t> newData;
    newData.resize(newWidth * newHeight * im.GetNumChannels() * im.GetSingleElementSizeInBytes());

    // boost::numeric_cast<>() is used for user-provided data (protecting about overflows).
    // static_cast<> is ok for internal data (assumes that, when internal data was originally provided by a user,
    // a boost::numeric_cast<>() handled the conversion).
    const int nW = boost::numeric_cast<int>(newWidth);
    const int nH = boost::numeric_cast<int>(newHeight);

    const int w = static_cast<int>(im.GetWidth());
    const int h = static_cast<int>(im.GetHeight());
    const int numChannels = static_cast<int>(im.GetNumChannels());

    const int res = stbir_resize_uint8(im.m_Data.data(), w, h, 0, newData.data(), nW, nH, 0, numChannels);
    if (res == 0)
    {
        throw InferenceTestImageResizeFailed("The resizing operation failed");
    }

    im.m_Data.swap(newData);
    im.m_Width = newWidth;
    im.m_Height = newHeight;
}

std::vector<float> InferenceTestImage::Resize(unsigned int newWidth,
                                              unsigned int newHeight,
                                              const armnn::CheckLocation& location,
                                              const ResizingMethods meth,
                                              const std::array<float, 3>& mean,
                                              const std::array<float, 3>& stddev)
{
    std::vector<float> out;
    if (newWidth == 0 || newHeight == 0)
    {
        throw InferenceTestImageResizeFailed(boost::str(boost::format("None of the dimensions passed to a resize "
            "operation can be zero. Requested width: %1%. Requested height: %2%.") % newWidth % newHeight));
    }

    switch (meth) {
        case ResizingMethods::STB:
        {
            StbResize(*this, newWidth, newHeight);
            break;
        }
        case ResizingMethods::BilinearAndNormalized:
        {
            out = ResizeBilinearAndNormalize(*this, newWidth, newHeight, mean, stddev);
            break;
        }
        default:
            throw InferenceTestImageResizeFailed(boost::str(
                boost::format("Unknown resizing method asked ArmNN only supports {STB, BilinearAndNormalized} %1%")
                              % location.AsString()));
    }
    return out;
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
