//
// Copyright © 2017,2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Gather.hpp"

#include <armnn/backends/WorkloadData.hpp>

#include <fmt/format.h>

namespace armnn
{

void Gather(const TensorInfo& paramsInfo,
            const TensorInfo& indicesInfo,
            const TensorInfo& outputInfo,
            Decoder<float>& params,
            const int32_t* indices,
            Encoder<float>& output,
            const int32_t axis_int)
{
    IgnoreUnused(outputInfo);

    const int paramsRank = static_cast<int>(paramsInfo.GetNumDimensions());
    if((axis_int < -1 * paramsRank) || (paramsRank <= axis_int))
    {
        throw InvalidArgumentException((fmt::format("Gather: Axis {} is not within [-{}, {}) range",
                                                    axis_int, paramsRank, paramsRank)));
    }
    const unsigned int axis = (axis_int < 0) ? static_cast<unsigned int>(paramsRank + axis_int)
                                             : static_cast<unsigned int>(axis_int);

    const TensorShape& paramsShape = paramsInfo.GetShape();

    // Product of all dimensions to the left side of the axis
    unsigned int paramsOuterProduct = 1;
    for (unsigned int i = 0; i < axis; ++i)
    {
        paramsOuterProduct *= paramsShape[i];
    }
    // Product of all dimensions to the right side of the axis
    unsigned int paramsInnerProduct = 1;
    for (unsigned int k = 1 + axis; k < paramsInfo.GetNumDimensions(); ++k)
    {
        paramsInnerProduct *= paramsShape[k];
    }

    unsigned int offset = 0;
    unsigned int outIndex = 0;
    for (unsigned int i = 0; i < paramsOuterProduct; ++i)
    {
        for (unsigned int j = 0; j < indicesInfo.GetNumElements(); ++j)
        {
            unsigned int index =
                (indices[j] < 0) ? static_cast<unsigned int>(static_cast<int>(paramsShape[axis]) + indices[j])
                                 : static_cast<unsigned int>(indices[j]);

            if (index >= paramsShape[axis])
            {
                throw InvalidArgumentException((fmt::format("Gather: index >= paramsShape[axis]: {} >= {}",
                                                            index, paramsShape[axis] )));
            }

            unsigned int startOffset = (paramsInnerProduct * index) + offset;
            unsigned int endOffset = startOffset + paramsInnerProduct;

            for (unsigned int k = startOffset; k < endOffset; ++k)
            {
                params[k];
                float outputValue = params.Get();
                output[outIndex];
                output.Set(outputValue);
                ++outIndex;
            }
        }
        offset += paramsShape[axis] * paramsInnerProduct;
    }

    if (outIndex != outputInfo.GetNumElements())
    {
        throw InvalidArgumentException((fmt::format("Gather: Invalid outIndex {} ", outIndex)));
    }
}

} //namespace armnn