//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <algorithm>
#include <cstring>
#include <boost/cast.hpp>
#include <Half.hpp>

#include "TensorCopyUtils.hpp"

#ifdef ARMCOMPUTECL_ENABLED
#include <backends/ClTensorHandle.hpp>
#endif

#if ARMCOMPUTENEON_ENABLED
#include <backends/neon/NeonTensorHandle.hpp>
#endif

#if ARMCOMPUTECLENABLED || ARMCOMPUTENEON_ENABLED
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>
#endif

#include "backends/CpuTensorHandle.hpp"

void CopyDataToITensorHandle(armnn::ITensorHandle* tensorHandle, const void* mem)
{
    switch (tensorHandle->GetType())
    {
        case armnn::ITensorHandle::Cpu:
        {
            auto handle = boost::polymorphic_downcast<armnn::ScopedCpuTensorHandle*>(tensorHandle);
            memcpy(handle->GetTensor<void>(), mem, handle->GetTensorInfo().GetNumBytes());
            break;
        }
#ifdef ARMCOMPUTECL_ENABLED
        case armnn::ITensorHandle::CL:
        {
            using armnn::armcomputetensorutils::CopyArmComputeITensorData;
            auto handle = boost::polymorphic_downcast<armnn::IClTensorHandle*>(tensorHandle);
            handle->Map(true);
            switch(handle->GetDataType())
            {
                case arm_compute::DataType::F32:
                    CopyArmComputeITensorData(static_cast<const float*>(mem), handle->GetTensor());
                    break;
                case arm_compute::DataType::QASYMM8:
                    CopyArmComputeITensorData(static_cast<const uint8_t*>(mem), handle->GetTensor());
                    break;
                case arm_compute::DataType::F16:
                    CopyArmComputeITensorData(static_cast<const armnn::Half*>(mem), handle->GetTensor());
                    break;
                default:
                {
                    throw armnn::UnimplementedException();
                }
            }
            handle->Unmap();
            break;
        }
#endif
#if ARMCOMPUTENEON_ENABLED
        case armnn::ITensorHandle::Neon:
        {
            using armnn::armcomputetensorutils::CopyArmComputeITensorData;
            auto handle = boost::polymorphic_downcast<armnn::INeonTensorHandle*>(tensorHandle);
            switch (handle->GetDataType())
            {
                case arm_compute::DataType::F32:
                    CopyArmComputeITensorData(static_cast<const float*>(mem), handle->GetTensor());
                    break;
                case arm_compute::DataType::QASYMM8:
                    CopyArmComputeITensorData(static_cast<const uint8_t*>(mem), handle->GetTensor());
                    break;
                default:
                {
                    throw armnn::UnimplementedException();
                }
            }
            break;
        }
#endif
        default:
        {
            throw armnn::UnimplementedException();
        }
    }
}

void CopyDataFromITensorHandle(void* mem, const armnn::ITensorHandle* tensorHandle)
{
    switch (tensorHandle->GetType())
    {
        case armnn::ITensorHandle::Cpu:
        {
            auto handle = boost::polymorphic_downcast<const armnn::ScopedCpuTensorHandle*>(tensorHandle);
            memcpy(mem, handle->GetTensor<void>(), handle->GetTensorInfo().GetNumBytes());
            break;
        }
#ifdef ARMCOMPUTECL_ENABLED
        case armnn::ITensorHandle::CL:
        {
            using armnn::armcomputetensorutils::CopyArmComputeITensorData;
            auto handle = boost::polymorphic_downcast<const armnn::IClTensorHandle*>(tensorHandle);
            const_cast<armnn::IClTensorHandle*>(handle)->Map(true);
            switch(handle->GetDataType())
            {
                case arm_compute::DataType::F32:
                    CopyArmComputeITensorData(handle->GetTensor(), static_cast<float*>(mem));
                    break;
                case arm_compute::DataType::QASYMM8:
                    CopyArmComputeITensorData(handle->GetTensor(), static_cast<uint8_t*>(mem));
                    break;
                case arm_compute::DataType::F16:
                    CopyArmComputeITensorData(handle->GetTensor(), static_cast<armnn::Half*>(mem));
                    break;
                default:
                {
                    throw armnn::UnimplementedException();
                }
            }
            const_cast<armnn::IClTensorHandle*>(handle)->Unmap();
            break;
        }
#endif
#if ARMCOMPUTENEON_ENABLED
        case armnn::ITensorHandle::Neon:
        {
            using armnn::armcomputetensorutils::CopyArmComputeITensorData;
            auto handle = boost::polymorphic_downcast<const armnn::INeonTensorHandle*>(tensorHandle);
            switch (handle->GetDataType())
            {
                case arm_compute::DataType::F32:
                    CopyArmComputeITensorData(handle->GetTensor(), static_cast<float*>(mem));
                    break;
                case arm_compute::DataType::QASYMM8:
                    CopyArmComputeITensorData(handle->GetTensor(), static_cast<uint8_t*>(mem));
                    break;
                default:
                {
                    throw armnn::UnimplementedException();
                }
            }
            break;
        }
#endif
        default:
        {
            throw armnn::UnimplementedException();
        }
    }
}

void AllocateAndCopyDataToITensorHandle(armnn::ITensorHandle* tensorHandle, const void* mem)
{
    tensorHandle->Allocate();
    CopyDataToITensorHandle(tensorHandle, mem);
}
