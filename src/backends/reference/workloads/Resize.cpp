//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Resize.hpp"

#include "TensorBufferArrayView.hpp"

#include <armnn/utility/NumericCast.hpp>

#include <cmath>
#include <algorithm>

using namespace armnnUtils;

namespace armnn
{

namespace
{

inline float Lerp(float a, float b, float w)
{
    return w * b + (1.f - w) * a;
}

inline double EuclideanDistance(float Xa, float Ya, const unsigned int Xb, const unsigned int Yb)
{
    return std::sqrt(pow(Xa - armnn::numeric_cast<float>(Xb), 2) + pow(Ya - armnn::numeric_cast<float>(Yb), 2));
}

inline float CalculateResizeScale(const unsigned int& InputSize,
                                  const unsigned int& OutputSize,
                                  const bool& AlignCorners)
{
    return (AlignCorners && OutputSize > 1)
            ?  armnn::numeric_cast<float>(InputSize - 1) / armnn::numeric_cast<float>(OutputSize - 1)
            :  armnn::numeric_cast<float>(InputSize) / armnn::numeric_cast<float>(OutputSize);
}

inline float PixelScaler(const unsigned int& Pixel,
                         const float& Scale,
                         const bool& HalfPixelCenters,
                         armnn::ResizeMethod& resizeMethod)
{
    // For Half Pixel Centers the Top Left texel is assumed to be at 0.5,0.5
    if (HalfPixelCenters && resizeMethod == armnn::ResizeMethod::Bilinear)
    {
        return (static_cast<float>(Pixel) + 0.5f) * Scale - 0.5f;
    }
    // Nearest Neighbour doesn't need to have 0.5f trimmed off as it will floor the values later
    else if (HalfPixelCenters && resizeMethod == armnn::ResizeMethod::NearestNeighbor)
    {
        return (static_cast<float>(Pixel) + 0.5f) * Scale;
    }
    else
    {
        return static_cast<float>(Pixel) * Scale;
    }
}

}// anonymous namespace

void Resize(Decoder<float>&   in,
            const TensorInfo& inputInfo,
            Encoder<float>&   out,
            const TensorInfo& outputInfo,
            DataLayoutIndexed dataLayout,
            armnn::ResizeMethod resizeMethod,
            bool alignCorners,
            bool halfPixelCenters)
{
    // alignCorners and halfPixelCenters cannot both be true
    ARMNN_ASSERT(!(alignCorners && halfPixelCenters));

    // We follow the definition of TensorFlow and AndroidNN: the top-left corner of a texel in the output
    // image is projected into the input image to figure out the interpolants and weights. Note that this
    // will yield different results than if projecting the centre of output texels.

    const unsigned int batchSize = inputInfo.GetShape()[0];
    const unsigned int channelCount = inputInfo.GetShape()[dataLayout.GetChannelsIndex()];

    const unsigned int inputHeight = inputInfo.GetShape()[dataLayout.GetHeightIndex()];
    const unsigned int inputWidth = inputInfo.GetShape()[dataLayout.GetWidthIndex()];
    const unsigned int outputHeight = outputInfo.GetShape()[dataLayout.GetHeightIndex()];
    const unsigned int outputWidth = outputInfo.GetShape()[dataLayout.GetWidthIndex()];

    // How much to scale pixel coordinates in the output image, to get the corresponding pixel coordinates
    // in the input image.
    const float scaleY = CalculateResizeScale(inputHeight, outputHeight, alignCorners);
    const float scaleX = CalculateResizeScale(inputWidth, outputWidth, alignCorners);

    TensorShape inputShape =  inputInfo.GetShape();
    TensorShape outputShape =  outputInfo.GetShape();

    for (unsigned int n = 0; n < batchSize; ++n)
    {
        for (unsigned int c = 0; c < channelCount; ++c)
        {
            for (unsigned int y = 0; y < outputHeight; ++y)
            {
                // Corresponding real-valued height coordinate in input image.
                float iy = PixelScaler(y, scaleY, halfPixelCenters, resizeMethod);

                // Discrete height coordinate of top-left texel (in the 2x2 texel area used for interpolation).
                const float fiy = (resizeMethod == armnn::ResizeMethod::NearestNeighbor && alignCorners) ?
                                  roundf(iy) : floorf(iy);
                // Pixel scaling a value with Half Pixel Centers can be negative, if so set to 0
                const unsigned int y0 = static_cast<unsigned int>(std::max(fiy, 0.0f));

                // Interpolation weight (range [0,1]).
                const float yw = iy - fiy;

                for (unsigned int x = 0; x < outputWidth; ++x)
                {
                    // Real-valued and discrete width coordinates in input image.
                    float ix = PixelScaler(x, scaleX, halfPixelCenters, resizeMethod);

                    // Nearest Neighbour uses rounding to align to corners
                    const float fix = resizeMethod == armnn::ResizeMethod::NearestNeighbor && alignCorners ?
                                      roundf(ix) : floorf(ix);
                    // Pixel scaling a value with Half Pixel Centers can be negative, if so set to 0
                    const unsigned int x0 = static_cast<unsigned int>(std::max(fix, 0.0f));

                    // Interpolation weight (range [0,1]).
                    const float xw = ix - fix;

                    unsigned int x1;
                    unsigned int y1;
                    // Half Pixel Centers uses the scaling to compute a weighted parameter for nearby pixels
                    if (halfPixelCenters)
                    {
                        x1 = std::min(static_cast<unsigned int>(std::ceil(ix)), inputWidth - 1u);
                        y1 = std::min(static_cast<unsigned int>(std::ceil(iy)), inputHeight - 1u);
                    }
                    // Discrete width/height coordinates of texels below and to the right of (x0, y0).
                    else
                    {
                        x1 = std::min(x0 + 1, inputWidth - 1u);
                        y1 = std::min(y0 + 1, inputHeight - 1u);
                    }

                    float interpolatedValue;
                    switch (resizeMethod)
                    {
                        case armnn::ResizeMethod::Bilinear:
                        {
                            in[dataLayout.GetIndex(inputShape, n, c, y0, x0)];
                            float input1 = in.Get();
                            in[dataLayout.GetIndex(inputShape, n, c, y0, x1)];
                            float input2 = in.Get();
                            in[dataLayout.GetIndex(inputShape, n, c, y1, x0)];
                            float input3 = in.Get();
                            in[dataLayout.GetIndex(inputShape, n, c, y1, x1)];
                            float input4 = in.Get();

                            const float ly0 = Lerp(input1, input2, xw); // lerp along row y0.
                            const float ly1 = Lerp(input3, input4, xw); // lerp along row y1.
                            interpolatedValue = Lerp(ly0, ly1, yw);
                            break;
                        }
                        case armnn::ResizeMethod::NearestNeighbor:
                        {
                            // calculate euclidean distance to the 4 neighbours
                            auto distance00 = EuclideanDistance(fix, fiy, x0, y0);
                            auto distance01 = EuclideanDistance(fix, fiy, x0, y1);
                            auto distance10 = EuclideanDistance(fix, fiy, x1, y0);
                            auto distance11 = EuclideanDistance(fix, fiy, x1, y1);

                            auto minimum = std::min( { distance00, distance01, distance10, distance11 } );

                            unsigned int xNearest = 0;
                            unsigned int yNearest = 0;

                            if (minimum == distance00)
                            {
                               xNearest = x0;
                               yNearest = y0;
                            }
                            else if (minimum == distance01)
                            {
                                xNearest = x0;
                                yNearest = y1;
                            }
                            else if (minimum == distance10)
                            {
                                xNearest = x1;
                                yNearest = y0;
                            }
                            else if (minimum == distance11)
                            {
                                xNearest = x1;
                                yNearest = y1;
                            }
                            else
                            {
                                throw armnn::InvalidArgumentException("Resize Nearest Neighbor failure");
                            }

                            in[dataLayout.GetIndex(inputShape, n, c, yNearest, xNearest)];
                            interpolatedValue = in.Get();
                            break;
                        }
                        default:
                            throw armnn::InvalidArgumentException("Unknown resize method: " +
                                                                  std::to_string(static_cast<int>(resizeMethod)));
                    }
                    out[dataLayout.GetIndex(outputShape, n, c, y, x)];
                    out.Set(interpolatedValue);
                }
            }
        }
    }
}

} //namespace armnn
