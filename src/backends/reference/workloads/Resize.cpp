//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Resize.hpp"

#include "TensorBufferArrayView.hpp"

#include <boost/numeric/conversion/cast.hpp>

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
    return std::sqrt(pow(Xa - boost::numeric_cast<float>(Xb), 2) + pow(Ya - boost::numeric_cast<float>(Yb), 2));
}

}// anonymous namespace

void Resize(Decoder<float>&   in,
            const TensorInfo& inputInfo,
            Encoder<float>&   out,
            const TensorInfo& outputInfo,
            DataLayoutIndexed dataLayout,
            armnn::ResizeMethod resizeMethod)
{
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
    const float scaleY = boost::numeric_cast<float>(inputHeight) / boost::numeric_cast<float>(outputHeight);
    const float scaleX = boost::numeric_cast<float>(inputWidth) / boost::numeric_cast<float>(outputWidth);

    TensorShape inputShape =  inputInfo.GetShape();
    TensorShape outputShape =  outputInfo.GetShape();

    for (unsigned int n = 0; n < batchSize; ++n)
    {
        for (unsigned int c = 0; c < channelCount; ++c)
        {
            for (unsigned int y = 0; y < outputHeight; ++y)
            {
                // Corresponding real-valued height coordinate in input image.
                const float iy = boost::numeric_cast<float>(y) * scaleY;

                // Discrete height coordinate of top-left texel (in the 2x2 texel area used for interpolation).
                const float fiy = floorf(iy);
                const unsigned int y0 = boost::numeric_cast<unsigned int>(fiy);

                // Interpolation weight (range [0,1]).
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
