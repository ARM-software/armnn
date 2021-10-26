//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Pooling3d.hpp"

#include <armnn/Exceptions.hpp>
#include <armnn/Types.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <limits>
#include <algorithm>
#include <functional>

namespace
{
    using PoolingAlgorithm = armnn::PoolingAlgorithm;

    float DefaultInitializer(PoolingAlgorithm algorithm)
    {
        switch (algorithm)
        {
            case PoolingAlgorithm::Max:
            {
                return std::numeric_limits<float>::lowest();
            }
            case PoolingAlgorithm::Average:
            case PoolingAlgorithm::L2:
            {
                return 0.0f;
            }
            default:
            {
                throw armnn::InvalidArgumentException("Unsupported pooling algorithm");
            }
        }
    }

    using Accumulator = std::function<void(float & accu, float value)>;

    Accumulator GetAccumulator(PoolingAlgorithm algorithm)
    {
        switch (algorithm)
        {
            case PoolingAlgorithm::Max:
            {
                return [](float & accu, float value) {
                    if (value > accu) {
                        accu = value;
                    }
                };
            }

            case PoolingAlgorithm::Average:
            {
                return [](float & accu, float value) {
                    accu += value;
                };
            }

            case PoolingAlgorithm::L2:
            {
                return [](float & accu, float value) {
                    accu += (value*value);
                };
            }

            default:
            {
                throw armnn::InvalidArgumentException("Unsupported pooling algorithm");
            }
        }
    }

    using Executor = std::function<void(float & accumulated, float kernelSize)>;

    Executor GetExecutor(PoolingAlgorithm algorithm)
    {
        switch (algorithm)
        {
            case PoolingAlgorithm::Max:
            {
                return [](float & /*accumulated*/, float /*kernelSize*/) {};
            }

            case PoolingAlgorithm::Average:
            {
                return [](float & accumulated, float kernelSize) {
                    accumulated /= kernelSize;
                };
            }

            case PoolingAlgorithm::L2:
            {
                return [](float & accumulated, float kernelSize) {
                    accumulated = sqrtf(accumulated / kernelSize);
                };
            }

            default:
            {
                throw armnn::InvalidArgumentException("Unsupported pooling algorithm");
            }
        }
    }

    bool OnPaddingOnly(int start, int end, int maxRange)
    {
        if (end <= 0 || start > maxRange)
        {
            return true;
        }
        else
        {
            return false;
        }
    }


    bool ClampRange(int & start, int & end, int maxRange)
    {
        if (start < 0 || end > maxRange)
        {
            start = std::min(std::max(start, 0), maxRange);
            end   = std::min(std::max(end, 0), maxRange);
            return true;
        }
        else
        {
            return false;
        }
    }

    int CalculateIndex(int channels, int depth, int height, int width,
                             int n, int c, int z, int y, int x,
                            armnnUtils::DataLayoutIndexed dataLayout) {
        switch (dataLayout.GetDataLayout())
        {
            case armnn::DataLayout::NDHWC:
            {
                int outputIndex = n * depth * height * width * channels +
                            z * height * width * channels +
                            y * width * channels +
                            x * channels +
                            c;
                return outputIndex;
            }
            case armnn::DataLayout::NCDHW:
            {
                int outputIndex = n * channels * depth * height * width +
                            c * depth * height * width +
                            z * height * width +
                            y * width +
                            x;
                return outputIndex;
            }
            default:
            {
                throw armnn::InvalidArgumentException("Unsupported data layout.");
            }
        }
    }
}

using namespace armnnUtils;

namespace armnn
{
void Pooling3d(Decoder<float>& rInputDecoder,
               Encoder<float>& rOutputEncoder,
               const TensorInfo& inputInfo,
               const TensorInfo& outputInfo,
               const Pooling3dDescriptor& params)
{
    const DataLayoutIndexed dataLayout(params.m_DataLayout);

    auto channelsIndex = dataLayout.GetChannelsIndex();

    auto depthIndex = dataLayout.GetDepthIndex();
    auto heightIndex = dataLayout.GetHeightIndex();
    auto widthIndex = dataLayout.GetWidthIndex();

    const int batchSize    = armnn::numeric_cast<int>(outputInfo.GetShape()[0]);
    const int channels     = armnn::numeric_cast<int>(outputInfo.GetShape()[channelsIndex]);

    const int depthOutput  = armnn::numeric_cast<int>(outputInfo.GetShape()[depthIndex]);
    const int heightOutput = armnn::numeric_cast<int>(outputInfo.GetShape()[heightIndex]);
    const int widthOutput  = armnn::numeric_cast<int>(outputInfo.GetShape()[widthIndex]);

    const int depthInput   = armnn::numeric_cast<int>(inputInfo.GetShape()[depthIndex]);
    const int heightInput  = armnn::numeric_cast<int>(inputInfo.GetShape()[heightIndex]);
    const int widthInput   = armnn::numeric_cast<int>(inputInfo.GetShape()[widthIndex]);

    const int padLeft      = armnn::numeric_cast<int>(params.m_PadLeft);
    const int padRight     = armnn::numeric_cast<int>(params.m_PadRight);
    const int padTop       = armnn::numeric_cast<int>(params.m_PadTop);
    const int padBottom    = armnn::numeric_cast<int>(params.m_PadBottom);
    const int padFront     = armnn::numeric_cast<int>(params.m_PadFront);
    const int padBack      = armnn::numeric_cast<int>(params.m_PadBack);

    const int strideX      = armnn::numeric_cast<int>(params.m_StrideX);
    const int strideY      = armnn::numeric_cast<int>(params.m_StrideY);
    const int strideZ      = armnn::numeric_cast<int>(params.m_StrideZ);

    const int poolHeight   = armnn::numeric_cast<int>(params.m_PoolHeight);
    const int poolWidth    = armnn::numeric_cast<int>(params.m_PoolWidth);
    const int poolDepth    = armnn::numeric_cast<int>(params.m_PoolDepth);

    float defaultInitializer = DefaultInitializer(params.m_PoolType);
    Accumulator accumulate = GetAccumulator(params.m_PoolType);
    Executor execute       = GetExecutor(params.m_PoolType);

    // Check supported padding methods outside the loop to simplify
    // the inner loop.
    if (params.m_PaddingMethod != PaddingMethod::Exclude &&
        params.m_PaddingMethod != PaddingMethod::IgnoreValue)
    {
        throw armnn::InvalidArgumentException("Unsupported padding type");
    }

    const std::vector<float> decodedInputVec = rInputDecoder.DecodeTensor(inputInfo.GetShape());

    for (int n = 0; n < batchSize; n++)
    {
        for (int c = 0; c < channels; c++)
        {
            for (int zOutput = 0; zOutput < depthOutput; zOutput++)
            {
                //  Calculate values independent of the x and y axis
                int dstart = (zOutput * strideZ) - padFront;
                int dend = dstart + poolDepth;
                // Clamp the pooling region inside the valid input area (which includes the padding).
                // This is necessary because the final pooling in a row may overlap beyond the padding.
                dend = std::min(dend, depthInput + padBack);

                int depth = dend - dstart;
                bool dclamped = ClampRange(dstart, dend, depthInput);
                int depthClamped = dend - dstart;

                for (int yOutput = 0; yOutput < heightOutput; yOutput++)
                {
                    int hstart = (yOutput * strideY) - padTop;
                    int hend = hstart + poolHeight;
                    // Clamp the pooling region inside the valid input area (which includes the padding).
                    // This is necessary because the final pooling in a row may overlap beyond the padding.
                    hend = std::min(hend, heightInput + padBottom);

                    int height = hend - hstart;
                    bool hclamped = ClampRange(hstart, hend, heightInput);
                    int heightClamped = hend - hstart;

                    for (int xOutput = 0; xOutput < widthOutput; xOutput++)
                    {
                        int wstart = (xOutput * strideX) - padLeft;
                        int wend = wstart + poolWidth;
                        // Clamp the pooling region inside the valid input area (which includes the padding).
                        // This is necessary because the final pooling in a row may overlap beyond the padding.
                        wend = std::min(wend, widthInput + padRight);

                        int width = wend - wstart;
                        bool wclamped = ClampRange(wstart, wend, widthInput);
                        int widthClamped = wend - wstart;

                        float result = defaultInitializer;
                        float poolAreaSize = armnn::numeric_cast<float>(depth * height * width);

                        // Special case: when the pooling kernel is over a padding region and the padding
                        //               size is larger or equal to the kernel and the kernel only covers
                        //               padding and no real values, then we initialize the result as zero
                        //               by convention. This is because we need to choose a value here and
                        //               all values we have are padding, which we ignore.
                        if (OnPaddingOnly(dstart, dend, depthInput) ||
                            OnPaddingOnly(hstart, hend, heightInput) ||
                            OnPaddingOnly(wstart, wend, widthInput))
                        {
                            result = 0.0f;

                            int outputIndex = CalculateIndex(channels, depthOutput, heightOutput, widthOutput,
                                n, c, zOutput, yOutput, xOutput, dataLayout);

                            rOutputEncoder[static_cast<unsigned int>(outputIndex)];
                            rOutputEncoder.Set(result);

                            continue;
                        }

                        bool clamped = (dclamped | hclamped | wclamped);

                        if (clamped && params.m_PaddingMethod == PaddingMethod::Exclude)
                        {
                            // When we exclude the padding, it means we calculate with a smaller
                            // kernel size, so I changed the divisor here.
                            poolAreaSize = armnn::numeric_cast<float>(depthClamped * heightClamped * widthClamped);
                        }

                        for (auto zInput = dstart; zInput < dend; zInput++)
                        {
                            for (auto yInput = hstart; yInput < hend; yInput++)
                            {
                                for (auto xInput = wstart; xInput < wend; xInput++)
                                {

                                    int inputIndex = CalculateIndex(channels, depthInput, heightInput, widthInput,
                                n, c, zInput, yInput, xInput, dataLayout);

                                    accumulate(result, decodedInputVec[static_cast<unsigned int>(inputIndex)]);
                                }
                            }
                        }

                        execute(result, poolAreaSize);

                        int outputIndex = CalculateIndex(channels, depthOutput, heightOutput, widthOutput,
                            n, c, zOutput, yOutput, xOutput, dataLayout);

                        rOutputEncoder[static_cast<unsigned int>(outputIndex)];
                        rOutputEncoder.Set(result);
                    }
                }
            }
        }
    }
}

} //namespace armnn
