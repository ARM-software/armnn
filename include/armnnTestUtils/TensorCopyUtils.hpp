//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Tensor.hpp>

#include <armnn/backends/ITensorHandle.hpp>

void CopyDataToITensorHandle(armnn::ITensorHandle* tensorHandle, const void* memory);

void CopyDataFromITensorHandle(void* mem, const armnn::ITensorHandle* tensorHandle);

void AllocateAndCopyDataToITensorHandle(armnn::ITensorHandle* tensorHandle, const void* memory);