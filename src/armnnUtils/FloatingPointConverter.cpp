//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "FloatingPointConverter.hpp"

#include "Half.hpp"

#include <boost/assert.hpp>

namespace armnnUtils
{

void FloatingPointConverter::ConvertFloat32To16(const float* srcFloat32Buffer,
                                                size_t numElements,
                                                void* dstFloat16Buffer)
{
    BOOST_ASSERT(srcFloat32Buffer != nullptr);
    BOOST_ASSERT(dstFloat16Buffer != nullptr);

    armnn::Half* pHalf = reinterpret_cast<armnn::Half*>(dstFloat16Buffer);

    for (size_t i = 0; i < numElements; i++)
    {
        pHalf[i] = armnn::Half(srcFloat32Buffer[i]);
    }
}

void FloatingPointConverter::ConvertFloat16To32(const void* srcFloat16Buffer,
                                                size_t numElements,
                                                float* dstFloat32Buffer)
{
    BOOST_ASSERT(srcFloat16Buffer != nullptr);
    BOOST_ASSERT(dstFloat32Buffer != nullptr);

    const armnn::Half* pHalf = reinterpret_cast<const armnn::Half*>(srcFloat16Buffer);

    for (size_t i = 0; i < numElements; i++)
    {
        dstFloat32Buffer[i] = pHalf[i];
    }
}

} //namespace armnnUtils
