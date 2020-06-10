//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnnUtils/FloatingPointConverter.hpp>

#include "BFloat16.hpp"
#include "Half.hpp"

#include <armnn/utility/Assert.hpp>

namespace armnnUtils
{

void FloatingPointConverter::ConvertFloat32To16(const float* srcFloat32Buffer,
                                                size_t numElements,
                                                void* dstFloat16Buffer)
{
    ARMNN_ASSERT(srcFloat32Buffer != nullptr);
    ARMNN_ASSERT(dstFloat16Buffer != nullptr);

    armnn::Half* pHalf = static_cast<armnn::Half*>(dstFloat16Buffer);

    for (size_t i = 0; i < numElements; i++)
    {
        pHalf[i] = armnn::Half(srcFloat32Buffer[i]);
    }
}

void FloatingPointConverter::ConvertFloat16To32(const void* srcFloat16Buffer,
                                                size_t numElements,
                                                float* dstFloat32Buffer)
{
    ARMNN_ASSERT(srcFloat16Buffer != nullptr);
    ARMNN_ASSERT(dstFloat32Buffer != nullptr);

    const armnn::Half* pHalf = static_cast<const armnn::Half*>(srcFloat16Buffer);

    for (size_t i = 0; i < numElements; i++)
    {
        dstFloat32Buffer[i] = pHalf[i];
    }
}

void FloatingPointConverter::ConvertFloat32ToBFloat16(const float* srcFloat32Buffer,
                                                      size_t numElements,
                                                      void* dstBFloat16Buffer)
{
    ARMNN_ASSERT(srcFloat32Buffer != nullptr);
    ARMNN_ASSERT(dstBFloat16Buffer != nullptr);

    armnn::BFloat16* bf16 = static_cast<armnn::BFloat16*>(dstBFloat16Buffer);

    for (size_t i = 0; i < numElements; i++)
    {
        bf16[i] = armnn::BFloat16(srcFloat32Buffer[i]);
    }
}

void FloatingPointConverter::ConvertBFloat16ToFloat32(const void* srcBFloat16Buffer,
                                                      size_t numElements,
                                                      float* dstFloat32Buffer)
{
    ARMNN_ASSERT(srcBFloat16Buffer != nullptr);
    ARMNN_ASSERT(dstFloat32Buffer != nullptr);

    const armnn::BFloat16* bf16 = static_cast<const armnn::BFloat16*>(srcBFloat16Buffer);

    for (size_t i = 0; i < numElements; i++)
    {
        dstFloat32Buffer[i] = bf16[i].ToFloat32();
    }
}

} //namespace armnnUtils
