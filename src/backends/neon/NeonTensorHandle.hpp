//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <BFloat16.hpp>
#include <Half.hpp>

#include <armnn/utility/Assert.hpp>

#include <aclCommon/ArmComputeTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <arm_compute/runtime/MemoryGroup.h>
#include <arm_compute/runtime/IMemoryGroup.h>
#include <arm_compute/runtime/Tensor.h>
#include <arm_compute/runtime/SubTensor.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Coordinates.h>

namespace armnn
{

class NeonTensorHandle : public IAclTensorHandle
{
public:
    NeonTensorHandle(const TensorInfo& tensorInfo)
                     : m_ImportFlags(static_cast<MemorySourceFlags>(MemorySource::Malloc)),
                       m_Imported(false),
                       m_IsImportEnabled(false),
                       m_TypeAlignment(GetDataTypeSize(tensorInfo.GetDataType()))
    {
        armnn::armcomputetensorutils::BuildArmComputeTensor(m_Tensor, tensorInfo);
    }

    NeonTensorHandle(const TensorInfo& tensorInfo,
                     DataLayout dataLayout,
                     MemorySourceFlags importFlags = static_cast<MemorySourceFlags>(MemorySource::Malloc))
                     : m_ImportFlags(importFlags),
                       m_Imported(false),
                       m_IsImportEnabled(false),
                       m_TypeAlignment(GetDataTypeSize(tensorInfo.GetDataType()))


    {
        armnn::armcomputetensorutils::BuildArmComputeTensor(m_Tensor, tensorInfo, dataLayout);
    }

    arm_compute::ITensor& GetTensor() override { return m_Tensor; }
    arm_compute::ITensor const& GetTensor() const override { return m_Tensor; }

    virtual void Allocate() override
    {
        // If we have enabled Importing, don't Allocate the tensor
        if (!m_IsImportEnabled)
        {
            armnn::armcomputetensorutils::InitialiseArmComputeTensorEmpty(m_Tensor);
        }
    };

    virtual void Manage() override
    {
        // If we have enabled Importing, don't manage the tensor
        if (!m_IsImportEnabled)
        {
            ARMNN_ASSERT(m_MemoryGroup != nullptr);
            m_MemoryGroup->manage(&m_Tensor);
        }
    }

    virtual ITensorHandle* GetParent() const override { return nullptr; }

    virtual arm_compute::DataType GetDataType() const override
    {
        return m_Tensor.info()->data_type();
    }

    virtual void SetMemoryGroup(const std::shared_ptr<arm_compute::IMemoryGroup>& memoryGroup) override
    {
        m_MemoryGroup = PolymorphicPointerDowncast<arm_compute::MemoryGroup>(memoryGroup);
    }

    virtual const void* Map(bool /* blocking = true */) const override
    {
        return static_cast<const void*>(m_Tensor.buffer() + m_Tensor.info()->offset_first_element_in_bytes());
    }

    virtual void Unmap() const override {}

    TensorShape GetStrides() const override
    {
        return armcomputetensorutils::GetStrides(m_Tensor.info()->strides_in_bytes());
    }

    TensorShape GetShape() const override
    {
        return armcomputetensorutils::GetShape(m_Tensor.info()->tensor_shape());
    }

    void SetImportFlags(MemorySourceFlags importFlags)
    {
        m_ImportFlags = importFlags;
    }

    MemorySourceFlags GetImportFlags() const override
    {
        return m_ImportFlags;
    }

    void SetImportEnabledFlag(bool importEnabledFlag)
    {
        m_IsImportEnabled = importEnabledFlag;
    }

    bool CanBeImported(void* memory, MemorySource source) override
    {
        armnn::IgnoreUnused(source);
        if (reinterpret_cast<uintptr_t>(memory) % m_TypeAlignment)
        {
            return false;
        }
        return true;
    }

    virtual bool Import(void* memory, MemorySource source) override
    {
        if (m_ImportFlags & static_cast<MemorySourceFlags>(source))
        {
            if (source == MemorySource::Malloc && m_IsImportEnabled)
            {
                if (!CanBeImported(memory, source))
                {
                    throw MemoryImportException("NeonTensorHandle::Import Attempting to import unaligned memory");
                }

                // m_Tensor not yet Allocated
                if (!m_Imported && !m_Tensor.buffer())
                {
                    arm_compute::Status status = m_Tensor.allocator()->import_memory(memory);
                    // Use the overloaded bool operator of Status to check if it worked, if not throw an exception
                    // with the Status error message
                    m_Imported = bool(status);
                    if (!m_Imported)
                    {
                        throw MemoryImportException(status.error_description());
                    }
                    return m_Imported;
                }

                // m_Tensor.buffer() initially allocated with Allocate().
                if (!m_Imported && m_Tensor.buffer())
                {
                    throw MemoryImportException(
                        "NeonTensorHandle::Import Attempting to import on an already allocated tensor");
                }

                // m_Tensor.buffer() previously imported.
                if (m_Imported)
                {
                    arm_compute::Status status = m_Tensor.allocator()->import_memory(memory);
                    // Use the overloaded bool operator of Status to check if it worked, if not throw an exception
                    // with the Status error message
                    m_Imported = bool(status);
                    if (!m_Imported)
                    {
                        throw MemoryImportException(status.error_description());
                    }
                    return m_Imported;
                }
            }
            else
            {
                throw MemoryImportException("NeonTensorHandle::Import is disabled");
            }
        }
        else
        {
            throw MemoryImportException("NeonTensorHandle::Incorrect import flag");
        }
        return false;
    }

private:
    // Only used for testing
    void CopyOutTo(void* memory) const override
    {
        switch (this->GetDataType())
        {
            case arm_compute::DataType::F32:
                armcomputetensorutils::CopyArmComputeITensorData(this->GetTensor(),
                                                                 static_cast<float*>(memory));
                break;
            case arm_compute::DataType::U8:
            case arm_compute::DataType::QASYMM8:
                armcomputetensorutils::CopyArmComputeITensorData(this->GetTensor(),
                                                                 static_cast<uint8_t*>(memory));
                break;
            case arm_compute::DataType::QSYMM8:
            case arm_compute::DataType::QASYMM8_SIGNED:
                armcomputetensorutils::CopyArmComputeITensorData(this->GetTensor(),
                                                                 static_cast<int8_t*>(memory));
                break;
            case arm_compute::DataType::BFLOAT16:
                armcomputetensorutils::CopyArmComputeITensorData(this->GetTensor(),
                                                                 static_cast<armnn::BFloat16*>(memory));
                break;
            case arm_compute::DataType::F16:
                armcomputetensorutils::CopyArmComputeITensorData(this->GetTensor(),
                                                                 static_cast<armnn::Half*>(memory));
                break;
            case arm_compute::DataType::S16:
            case arm_compute::DataType::QSYMM16:
                armcomputetensorutils::CopyArmComputeITensorData(this->GetTensor(),
                                                                 static_cast<int16_t*>(memory));
                break;
            case arm_compute::DataType::S32:
                armcomputetensorutils::CopyArmComputeITensorData(this->GetTensor(),
                                                                 static_cast<int32_t*>(memory));
                break;
            default:
            {
                throw armnn::UnimplementedException();
            }
        }
    }

    // Only used for testing
    void CopyInFrom(const void* memory) override
    {
        switch (this->GetDataType())
        {
            case arm_compute::DataType::F32:
                armcomputetensorutils::CopyArmComputeITensorData(static_cast<const float*>(memory),
                                                                 this->GetTensor());
                break;
            case arm_compute::DataType::U8:
            case arm_compute::DataType::QASYMM8:
                armcomputetensorutils::CopyArmComputeITensorData(static_cast<const uint8_t*>(memory),
                                                                 this->GetTensor());
                break;
            case arm_compute::DataType::QSYMM8:
            case arm_compute::DataType::QASYMM8_SIGNED:
                armcomputetensorutils::CopyArmComputeITensorData(static_cast<const int8_t*>(memory),
                                                                 this->GetTensor());
                break;
            case arm_compute::DataType::BFLOAT16:
                armcomputetensorutils::CopyArmComputeITensorData(static_cast<const armnn::BFloat16*>(memory),
                                                                 this->GetTensor());
                break;
            case arm_compute::DataType::F16:
                armcomputetensorutils::CopyArmComputeITensorData(static_cast<const armnn::Half*>(memory),
                                                                 this->GetTensor());
                break;
            case arm_compute::DataType::S16:
            case arm_compute::DataType::QSYMM16:
                armcomputetensorutils::CopyArmComputeITensorData(static_cast<const int16_t*>(memory),
                                                                 this->GetTensor());
                break;
            case arm_compute::DataType::S32:
                armcomputetensorutils::CopyArmComputeITensorData(static_cast<const int32_t*>(memory),
                                                                 this->GetTensor());
                break;
            default:
            {
                throw armnn::UnimplementedException();
            }
        }
    }

    arm_compute::Tensor m_Tensor;
    std::shared_ptr<arm_compute::MemoryGroup> m_MemoryGroup;
    MemorySourceFlags m_ImportFlags;
    bool m_Imported;
    bool m_IsImportEnabled;
    const uintptr_t m_TypeAlignment;
};

class NeonSubTensorHandle : public IAclTensorHandle
{
public:
    NeonSubTensorHandle(IAclTensorHandle* parent,
                        const arm_compute::TensorShape& shape,
                        const arm_compute::Coordinates& coords)
     : m_Tensor(&parent->GetTensor(), shape, coords)
    {
        parentHandle = parent;
    }

    arm_compute::ITensor& GetTensor() override { return m_Tensor; }
    arm_compute::ITensor const& GetTensor() const override { return m_Tensor; }

    virtual void Allocate() override {}
    virtual void Manage() override {}

    virtual ITensorHandle* GetParent() const override { return parentHandle; }

    virtual arm_compute::DataType GetDataType() const override
    {
        return m_Tensor.info()->data_type();
    }

    virtual void SetMemoryGroup(const std::shared_ptr<arm_compute::IMemoryGroup>&) override {}

    virtual const void* Map(bool /* blocking = true */) const override
    {
        return static_cast<const void*>(m_Tensor.buffer() + m_Tensor.info()->offset_first_element_in_bytes());
    }
    virtual void Unmap() const override {}

    TensorShape GetStrides() const override
    {
        return armcomputetensorutils::GetStrides(m_Tensor.info()->strides_in_bytes());
    }

    TensorShape GetShape() const override
    {
        return armcomputetensorutils::GetShape(m_Tensor.info()->tensor_shape());
    }

private:
    // Only used for testing
    void CopyOutTo(void* memory) const override
    {
        switch (this->GetDataType())
        {
            case arm_compute::DataType::F32:
                armcomputetensorutils::CopyArmComputeITensorData(this->GetTensor(),
                                                                 static_cast<float*>(memory));
                break;
            case arm_compute::DataType::U8:
            case arm_compute::DataType::QASYMM8:
                armcomputetensorutils::CopyArmComputeITensorData(this->GetTensor(),
                                                                 static_cast<uint8_t*>(memory));
                break;
            case arm_compute::DataType::QSYMM8:
            case arm_compute::DataType::QASYMM8_SIGNED:
                armcomputetensorutils::CopyArmComputeITensorData(this->GetTensor(),
                                                                 static_cast<int8_t*>(memory));
                break;
            case arm_compute::DataType::S16:
            case arm_compute::DataType::QSYMM16:
                armcomputetensorutils::CopyArmComputeITensorData(this->GetTensor(),
                                                                 static_cast<int16_t*>(memory));
                break;
            case arm_compute::DataType::S32:
                armcomputetensorutils::CopyArmComputeITensorData(this->GetTensor(),
                                                                 static_cast<int32_t*>(memory));
                break;
            default:
            {
                throw armnn::UnimplementedException();
            }
        }
    }

    // Only used for testing
    void CopyInFrom(const void* memory) override
    {
        switch (this->GetDataType())
        {
            case arm_compute::DataType::F32:
                armcomputetensorutils::CopyArmComputeITensorData(static_cast<const float*>(memory),
                                                                 this->GetTensor());
                break;
            case arm_compute::DataType::U8:
            case arm_compute::DataType::QASYMM8:
                armcomputetensorutils::CopyArmComputeITensorData(static_cast<const uint8_t*>(memory),
                                                                 this->GetTensor());
                break;
            case arm_compute::DataType::QSYMM8:
            case arm_compute::DataType::QASYMM8_SIGNED:
                armcomputetensorutils::CopyArmComputeITensorData(static_cast<const int8_t*>(memory),
                                                                 this->GetTensor());
                break;
            case arm_compute::DataType::S16:
            case arm_compute::DataType::QSYMM16:
                armcomputetensorutils::CopyArmComputeITensorData(static_cast<const int16_t*>(memory),
                                                                 this->GetTensor());
                break;
            case arm_compute::DataType::S32:
                armcomputetensorutils::CopyArmComputeITensorData(static_cast<const int32_t*>(memory),
                                                                 this->GetTensor());
                break;
            default:
            {
                throw armnn::UnimplementedException();
            }
        }
    }

    arm_compute::SubTensor m_Tensor;
    ITensorHandle* parentHandle = nullptr;
};

} // namespace armnn
