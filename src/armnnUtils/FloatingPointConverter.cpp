//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnnUtils/FloatingPointConverter.hpp>

#include "BFloat16.hpp"
#include "Half.hpp"

#include <armnn/Exceptions.hpp>
#include <armnn/utility/Assert.hpp>

namespace armnnUtils
{

void FloatingPointConverter::ConvertFloat32To16(const float* srcFloat32Buffer,
                                                size_t numElements,
                                                void* dstFloat16Buffer)
{
    if (srcFloat32Buffer == nullptr)
    {
        throw armnn::InvalidArgumentException("ConvertFloat32To16: source float32 buffer pointer is null");
    }
    if (dstFloat16Buffer == nullptr)
    {
        throw armnn::InvalidArgumentException("ConvertFloat32To16: destination float16 buffer pointer is null");
    }

    armnn::Half* pHalf = static_cast<armnn::Half*>(dstFloat16Buffer);

    for (size_t i = 0; i < numElements; i++)
    {
        pHalf[i] = armnn::Half(srcFloat32Buffer[i]);
        if (isinf(pHalf[i]))
        {
            // If the value of converted Fp16 is infinity, round to the closest finite Fp16 value.
            pHalf[i] = copysign(std::numeric_limits<armnn::Half>::max(), pHalf[i]);
        }
    }
}

void FloatingPointConverter::ConvertFloat16To32(const void* srcFloat16Buffer,
                                                size_t numElements,
                                                float* dstFloat32Buffer)
{
    if (srcFloat16Buffer == nullptr)
    {
        throw armnn::InvalidArgumentException("ConvertFloat16To32: source float16 buffer pointer is null");
    }
    if (dstFloat32Buffer == nullptr)
    {
        throw armnn::InvalidArgumentException("ConvertFloat16To32: destination float32 buffer pointer is null");
    }

    const armnn::Half* pHalf = static_cast<const armnn::Half*>(srcFloat16Buffer);

    for (size_t i = 0; i < numElements; i++)
    {
        dstFloat32Buffer[i] = pHalf[i];
    }
}

} //namespace armnnUtils
