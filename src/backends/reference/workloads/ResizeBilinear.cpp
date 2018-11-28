//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ResizeBilinear.hpp"

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

}

void ResizeBilinear(const float*      in,
                    const TensorInfo& inputInfo,
                    float*            out,
                    const TensorInfo& outputInfo,
                    DataLayoutIndexed dataLayout)
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

    TensorBufferArrayView<const float> input(inputInfo.GetShape(), in, dataLayout);
    TensorBufferArrayView<float> output(outputInfo.GetShape(), out, dataLayout);

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

                    // Interpolation
                    const float ly0 = Lerp(input.Get(n, c, y0, x0), input.Get(n, c, y0, x1), xw); // lerp along row y0.
                    const float ly1 = Lerp(input.Get(n, c, y1, x0), input.Get(n, c, y1, x1), xw); // lerp along row y1.
                    const float l = Lerp(ly0, ly1, yw);

                    output.Get(n, c, y, x) = l;
                }
            }
        }
    }
}

} //namespace armnn
