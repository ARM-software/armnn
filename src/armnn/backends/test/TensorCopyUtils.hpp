//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "armnn/Tensor.hpp"
#include "backends/ITensorHandle.hpp"

void CopyDataToITensorHandle(armnn::ITensorHandle* tensorHandle, const void* mem);

void CopyDataFromITensorHandle(void* mem, const armnn::ITensorHandle* tensorHandle);

void AllocateAndCopyDataToITensorHandle(armnn::ITensorHandle* tensorHandle, const void* mem);