//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TensorCopyUtils.hpp"
#include <Half.hpp>

void CopyDataToITensorHandle(armnn::ITensorHandle* tensorHandle, const void* memory)
{
    tensorHandle->CopyInFrom(memory);
}

void CopyDataFromITensorHandle(void* memory, const armnn::ITensorHandle* tensorHandle)
{
    tensorHandle->CopyOutTo(memory);
}

void AllocateAndCopyDataToITensorHandle(armnn::ITensorHandle* tensorHandle, const void* memory)
{
    tensorHandle->Allocate();
    CopyDataToITensorHandle(tensorHandle, memory);
}
