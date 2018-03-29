//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "InferenceTestImage.hpp"
#include "MobileNetDatabase.hpp"

#include <boost/numeric/conversion/cast.hpp>
#include <boost/assert.hpp>
#include <boost/format.hpp>

#include <iostream>
#include <fcntl.h>
#include <array>

namespace
{

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

std::vector<float>
ResizeBilinearAndNormalize(const InferenceTestImage & image,
                           const unsigned int outputWidth,
                           const unsigned int outputHeight)
{
    std::vector<float> out;
    out.resize(outputWidth * outputHeight * 3);

    // We follow the definition of TensorFlow and AndroidNN: The top-left corner of a texel in the output
    // image is projected into the input image to figure out the interpolants and weights. Note that this
    // will yield different results than if projecting the centre of output texels.

    const unsigned int inputWidth = image.GetWidth();
    const unsigned int inputHeight = image.GetHeight();

    // How much to scale pixel coordinates in the output image to get the corresponding pixel coordinates
    // in the input image
    const float scaleY = boost::numeric_cast<float>(inputHeight) / boost::numeric_cast<float>(outputHeight);
    const float scaleX = boost::numeric_cast<float>(inputWidth) / boost::numeric_cast<float>(outputWidth);

    uint8_t rgb_x0y0[3];
    uint8_t rgb_x1y0[3];
    uint8_t rgb_x0y1[3];
    uint8_t rgb_x1y1[3];

    for (unsigned int y = 0; y < outputHeight; ++y)
    {
        // Corresponding real-valued height coordinate in input image
        const float iy = boost::numeric_cast<float>(y) * scaleY;

        // Discrete height coordinate of top-left texel (in the 2x2 texel area used for interpolation)
        const float fiy = floorf(iy);
        const unsigned int y0 = boost::numeric_cast<unsigned int>(fiy);

        // Interpolation weight (range [0,1])
        const float yw = iy - fiy;

        for (unsigned int x = 0; x < outputWidth; ++x)
        {
            // Real-valued and discrete width coordinates in input image
            const float ix = boost::numeric_cast<float>(x) * scaleX;
            const float fix = floorf(ix);
            const unsigned int x0 = boost::numeric_cast<unsigned int>(fix);

            // Interpolation weight (range [0,1])
            const float xw = ix - fix;

            // Discrete width/height coordinates of texels below and to the right of (x0, y0)
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
                PutData(out, outputWidth, x, y, c, l/255.0f);
            }
        }
    }

    return out;
}

} // end of anonymous namespace


MobileNetDatabase::MobileNetDatabase(const std::string& binaryFileDirectory,
                                     unsigned int width,
                                     unsigned int height,
                                     const std::vector<ImageSet>& imageSet)
:   m_BinaryDirectory(binaryFileDirectory)
,   m_Height(height)
,   m_Width(width)
,   m_ImageSet(imageSet)
{
}

std::unique_ptr<MobileNetDatabase::TTestCaseData>
MobileNetDatabase::GetTestCaseData(unsigned int testCaseId)
{
    testCaseId = testCaseId % boost::numeric_cast<unsigned int>(m_ImageSet.size());
    const ImageSet& imageSet = m_ImageSet[testCaseId];
    const std::string fullPath = m_BinaryDirectory + imageSet.first;

    InferenceTestImage image(fullPath.c_str());

    // this ResizeBilinear result is closer to the tensorflow one than STB.
    // there is still some difference though, but the inference results are
    // similar to tensorflow for MobileNet
    std::vector<float> resized(ResizeBilinearAndNormalize(image, m_Width, m_Height));

    const unsigned int label = imageSet.second;
    return std::make_unique<TTestCaseData>(label, std::move(resized));
}
