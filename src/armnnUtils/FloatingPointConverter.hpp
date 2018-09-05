//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>

namespace armnnUtils
{
class FloatingPointConverter
{
public:
    // Converts a buffer of FP32 values to FP16, and stores in the given dstFloat16Buffer.
    // dstFloat16Buffer should be (numElements * 2) in size
    static void ConvertFloat32To16(const float *srcFloat32Buffer, size_t numElements, void *dstFloat16Buffer);

    static void ConvertFloat16To32(const void *srcFloat16Buffer, size_t numElements, float *dstFloat32Buffer);
};
} //namespace armnnUtils
