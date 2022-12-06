//
// Copyright © 2017,2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

namespace armnn
{
class ITensorHandle;
}  // namespace armnn

void CopyDataToITensorHandle(armnn::ITensorHandle* tensorHandle, const void* memory);

void CopyDataFromITensorHandle(void* mem, const armnn::ITensorHandle* tensorHandle);

void AllocateAndCopyDataToITensorHandle(armnn::ITensorHandle* tensorHandle, const void* memory);