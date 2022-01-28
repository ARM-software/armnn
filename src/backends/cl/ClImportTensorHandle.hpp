//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <aclCommon/ArmComputeTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <Half.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <arm_compute/runtime/CL/CLTensor.h>
#include <arm_compute/runtime/CL/CLSubTensor.h>
#include <arm_compute/runtime/IMemoryGroup.h>
#include <arm_compute/runtime/MemoryGroup.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Coordinates.h>

#include <cl/IClTensorHandle.hpp>

#include <CL/cl_ext.h>
#include <arm_compute/core/CL/CLKernelLibrary.h>

namespace armnn
{

class ClImportTensorHandle : public IClTensorHandle
{
public:
    ClImportTensorHandle(const TensorInfo& tensorInfo, MemorySourceFlags importFlags)
        : m_ImportFlags(importFlags)
    {
        armnn::armcomputetensorutils::BuildArmComputeTensor(m_Tensor, tensorInfo);
    }

    ClImportTensorHandle(const TensorInfo& tensorInfo,
                         DataLayout dataLayout,
                         MemorySourceFlags importFlags)
        : m_ImportFlags(importFlags), m_Imported(false)
    {
        armnn::armcomputetensorutils::BuildArmComputeTensor(m_Tensor, tensorInfo, dataLayout);
    }

    arm_compute::CLTensor& GetTensor() override { return m_Tensor; }
    arm_compute::CLTensor const& GetTensor() const override { return m_Tensor; }
    virtual void Allocate() override {}
    virtual void Manage() override {}

    virtual const void* Map(bool blocking = true) const override
    {
        IgnoreUnused(blocking);
        return static_cast<const void*>(m_Tensor.buffer() + m_Tensor.info()->offset_first_element_in_bytes());
    }

    virtual void Unmap() const override {}

    virtual ITensorHandle* GetParent() const override { return nullptr; }

    virtual arm_compute::DataType GetDataType() const override
    {
        return m_Tensor.info()->data_type();
    }

    virtual void SetMemoryGroup(const std::shared_ptr<arm_compute::IMemoryGroup>& memoryGroup) override
    {
        IgnoreUnused(memoryGroup);
    }

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

    virtual bool Import(void* memory, MemorySource source) override
    {
        if (m_ImportFlags & static_cast<MemorySourceFlags>(source))
        {
            if (source == MemorySource::Malloc)
            {
                const cl_import_properties_arm importProperties[] =
                {
                    CL_IMPORT_TYPE_ARM,
                    CL_IMPORT_TYPE_HOST_ARM,
                    0
                };

                return ClImport(importProperties, memory);
            }
            if (source == MemorySource::DmaBuf)
            {
                const cl_import_properties_arm importProperties[] =
                {
                    CL_IMPORT_TYPE_ARM,
                    CL_IMPORT_TYPE_DMA_BUF_ARM,
                    CL_IMPORT_DMA_BUF_DATA_CONSISTENCY_WITH_HOST_ARM,
                    CL_TRUE,
                    0
                };

                return ClImport(importProperties, memory);

            }
            if (source == MemorySource::DmaBufProtected)
            {
                const cl_import_properties_arm importProperties[] =
                {
                    CL_IMPORT_TYPE_ARM,
                    CL_IMPORT_TYPE_DMA_BUF_ARM,
                    CL_IMPORT_TYPE_PROTECTED_ARM,
                    CL_TRUE,
                    0
                };

                return ClImport(importProperties, memory, true);

            }
            // Case for importing memory allocated by OpenCl externally directly into the tensor
            else if (source == MemorySource::Gralloc)
            {
                // m_Tensor not yet Allocated
                if (!m_Imported && !m_Tensor.buffer())
                {
                    // Importing memory allocated by OpenCl into the tensor directly.
                    arm_compute::Status status =
                        m_Tensor.allocator()->import_memory(cl::Buffer(static_cast<cl_mem>(memory)));
                    m_Imported = bool(status);
                    if (!m_Imported)
                    {
                        throw MemoryImportException(status.error_description());
                    }
                    return m_Imported;
                }

                // m_Tensor.buffer() initially allocated with Allocate().
                else if (!m_Imported && m_Tensor.buffer())
                {
                    throw MemoryImportException(
                        "ClImportTensorHandle::Import Attempting to import on an already allocated tensor");
                }

                // m_Tensor.buffer() previously imported.
                else if (m_Imported)
                {
                    // Importing memory allocated by OpenCl into the tensor directly.
                    arm_compute::Status status =
                        m_Tensor.allocator()->import_memory(cl::Buffer(static_cast<cl_mem>(memory)));
                    m_Imported = bool(status);
                    if (!m_Imported)
                    {
                        throw MemoryImportException(status.error_description());
                    }
                    return m_Imported;
                }
                else
                {
                    throw MemoryImportException("ClImportTensorHandle::Failed to Import Gralloc Memory");
                }
            }
            else
            {
                throw MemoryImportException("ClImportTensorHandle::Import flag is not supported");
            }
        }
        else
        {
            throw MemoryImportException("ClImportTensorHandle::Incorrect import flag");
        }
    }

    virtual bool CanBeImported(void* memory, MemorySource source) override
    {
        if (m_ImportFlags & static_cast<MemorySourceFlags>(source))
        {
            if (source == MemorySource::Malloc)
            {
                const cl_import_properties_arm importProperties[] =
                        {
                                CL_IMPORT_TYPE_ARM,
                                CL_IMPORT_TYPE_HOST_ARM,
                                0
                        };

                size_t totalBytes = m_Tensor.info()->total_size();

                // Round the size of the mapping to match the CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE
                // This does not change the size of the buffer, only the size of the mapping the buffer is mapped to
                // We do this to match the behaviour of the Import function later on.
                auto cachelineAlignment =
                        arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
                auto roundedSize = totalBytes;
                if (totalBytes % cachelineAlignment != 0)
                {
                    roundedSize = cachelineAlignment + totalBytes - (totalBytes % cachelineAlignment);
                }

                cl_int error = CL_SUCCESS;
                cl_mem buffer;
                buffer = clImportMemoryARM(arm_compute::CLKernelLibrary::get().context().get(),
                                           CL_MEM_READ_WRITE, importProperties, memory, roundedSize, &error);

                // If we fail to map we know the import will not succeed and can return false.
                // There is no memory to be released if error is not CL_SUCCESS
                if (error != CL_SUCCESS)
                {
                    return false;
                }
                else
                {
                    // If import was successful we can release the mapping knowing import will succeed at workload
                    // execution and return true
                    error = clReleaseMemObject(buffer);
                    if (error == CL_SUCCESS)
                    {
                        return true;
                    }
                    else
                    {
                        // If we couldn't release the mapping this constitutes a memory leak and throw an exception
                        throw MemoryImportException("ClImportTensorHandle::Failed to unmap cl_mem buffer: "
                                                    + std::to_string(error));
                    }
                }
            }
        }
        else
        {
            throw MemoryImportException("ClImportTensorHandle::Incorrect import flag");
        }
        return false;
    }

private:
    bool ClImport(const cl_import_properties_arm* importProperties, void* memory, bool isProtected = false)
    {
        size_t totalBytes = m_Tensor.info()->total_size();

        // Round the size of the mapping to match the CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE
        // This does not change the size of the buffer, only the size of the mapping the buffer is mapped to
        auto cachelineAlignment =
                arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
        auto roundedSize = totalBytes;
        if (totalBytes % cachelineAlignment != 0)
        {
            roundedSize = cachelineAlignment + totalBytes - (totalBytes % cachelineAlignment);
        }

        cl_int error = CL_SUCCESS;
        cl_mem buffer;
        if (isProtected)
        {
            buffer = clImportMemoryARM(arm_compute::CLKernelLibrary::get().context().get(),
                                       CL_MEM_HOST_NO_ACCESS, importProperties, memory, roundedSize, &error);
        }
        else
        {
            buffer = clImportMemoryARM(arm_compute::CLKernelLibrary::get().context().get(),
                                       CL_MEM_READ_WRITE, importProperties, memory, roundedSize, &error);
        }

        if (error != CL_SUCCESS)
        {
            throw MemoryImportException("ClImportTensorHandle::Invalid imported memory" + std::to_string(error));
        }

        cl::Buffer wrappedBuffer(buffer);
        arm_compute::Status status = m_Tensor.allocator()->import_memory(wrappedBuffer);

        // Use the overloaded bool operator of Status to check if it is success, if not throw an exception
        // with the Status error message
        bool imported = (status.error_code() == arm_compute::ErrorCode::OK);
        if (!imported)
        {
            throw MemoryImportException(status.error_description());
        }

        ARMNN_ASSERT(!m_Tensor.info()->is_resizable());
        return imported;
    }
    // Only used for testing
    void CopyOutTo(void* memory) const override
    {
        const_cast<armnn::ClImportTensorHandle*>(this)->Map(true);
        switch(this->GetDataType())
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
            case arm_compute::DataType::QSYMM8_PER_CHANNEL:
            case arm_compute::DataType::QASYMM8_SIGNED:
                armcomputetensorutils::CopyArmComputeITensorData(this->GetTensor(),
                                                                 static_cast<int8_t*>(memory));
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
        const_cast<armnn::ClImportTensorHandle*>(this)->Unmap();
    }

    // Only used for testing
    void CopyInFrom(const void* memory) override
    {
        this->Map(true);
        switch(this->GetDataType())
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
            case arm_compute::DataType::F16:
                armcomputetensorutils::CopyArmComputeITensorData(static_cast<const armnn::Half*>(memory),
                                                                 this->GetTensor());
                break;
            case arm_compute::DataType::S16:
            case arm_compute::DataType::QSYMM8_PER_CHANNEL:
            case arm_compute::DataType::QASYMM8_SIGNED:
                armcomputetensorutils::CopyArmComputeITensorData(static_cast<const int8_t*>(memory),
                                                                 this->GetTensor());
                break;
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
        this->Unmap();
    }

    arm_compute::CLTensor m_Tensor;
    MemorySourceFlags m_ImportFlags;
    bool m_Imported;
};

class ClImportSubTensorHandle : public IClTensorHandle
{
public:
    ClImportSubTensorHandle(IClTensorHandle* parent,
                            const arm_compute::TensorShape& shape,
                            const arm_compute::Coordinates& coords)
    : m_Tensor(&parent->GetTensor(), shape, coords)
    {
        parentHandle = parent;
    }

    arm_compute::CLSubTensor& GetTensor() override { return m_Tensor; }
    arm_compute::CLSubTensor const& GetTensor() const override { return m_Tensor; }

    virtual void Allocate() override {}
    virtual void Manage() override {}

    virtual const void* Map(bool blocking = true) const override
    {
        IgnoreUnused(blocking);
        return static_cast<const void*>(m_Tensor.buffer() + m_Tensor.info()->offset_first_element_in_bytes());
    }
    virtual void Unmap() const override {}

    virtual ITensorHandle* GetParent() const override { return parentHandle; }

    virtual arm_compute::DataType GetDataType() const override
    {
        return m_Tensor.info()->data_type();
    }

    virtual void SetMemoryGroup(const std::shared_ptr<arm_compute::IMemoryGroup>& memoryGroup) override
    {
        IgnoreUnused(memoryGroup);
    }

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
        const_cast<ClImportSubTensorHandle*>(this)->Map(true);
        switch(this->GetDataType())
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
            case arm_compute::DataType::F16:
                armcomputetensorutils::CopyArmComputeITensorData(this->GetTensor(),
                                                                 static_cast<armnn::Half*>(memory));
                break;
            case arm_compute::DataType::QSYMM8_PER_CHANNEL:
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
        const_cast<ClImportSubTensorHandle*>(this)->Unmap();
    }

    // Only used for testing
    void CopyInFrom(const void* memory) override
    {
        this->Map(true);
        switch(this->GetDataType())
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
            case arm_compute::DataType::F16:
                armcomputetensorutils::CopyArmComputeITensorData(static_cast<const armnn::Half*>(memory),
                                                                 this->GetTensor());
                break;
            case arm_compute::DataType::QSYMM8_PER_CHANNEL:
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
        this->Unmap();
    }

    mutable arm_compute::CLSubTensor m_Tensor;
    ITensorHandle* parentHandle = nullptr;
};

} // namespace armnn
