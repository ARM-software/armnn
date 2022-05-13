//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IBackendInternal.hpp>

#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/CL/CLBufferAllocator.h>

#include <aclCommon/BaseMemoryManager.hpp>
#include <arm_compute/runtime/CL/CLMemoryRegion.h>

#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <CL/cl_ext.h>

// System includes for mapping and unmapping memory
#include <sys/mman.h>

namespace armnn
{

// add new capabilities here..
const BackendCapabilities gpuAccCapabilities("GpuAcc",
                                             {
                                                     {"NonConstWeights", false},
                                                     {"AsyncExecution", false},
                                                     {"ProtectedContentAllocation", true},
                                                     {"ConstantTensorsAsInputs", true},
                                                     {"PreImportIOTensors", false},
                                                     {"ExternallyManagedMemory", true},
                                                     {"MultiAxisPacking", false},
                                                     {"SingleAxisPacking", true}
                                             });

class ClBackend : public IBackendInternal
{
public:
    ClBackend() : m_CustomAllocator(nullptr) {};
    ClBackend(std::shared_ptr<ICustomAllocator> allocator)
    {
        std::string err;
        UseCustomMemoryAllocator(allocator, err);
    }
    ~ClBackend() = default;

    static const BackendId& GetIdStatic();
    const BackendId& GetId() const override { return GetIdStatic(); }

    IBackendInternal::IMemoryManagerUniquePtr CreateMemoryManager() const override;

    IBackendInternal::IWorkloadFactoryPtr CreateWorkloadFactory(
        const IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr) const override;

    IBackendInternal::IWorkloadFactoryPtr CreateWorkloadFactory(
        TensorHandleFactoryRegistry& registry) const override;

    IWorkloadFactoryPtr CreateWorkloadFactory(const IMemoryManagerSharedPtr& memoryManager,
                                              const ModelOptions& modelOptions) const override;

    IWorkloadFactoryPtr CreateWorkloadFactory(class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry,
                                              const ModelOptions& modelOptions) const override;

    IWorkloadFactoryPtr CreateWorkloadFactory(class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry,
                                              const ModelOptions& modelOptions,
                                              MemorySourceFlags inputFlags,
                                              MemorySourceFlags outputFlags) const override;

    std::vector<ITensorHandleFactory::FactoryId> GetHandleFactoryPreferences() const override;

    void RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry) override;

    void RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry,
                                       MemorySourceFlags inputFlags,
                                       MemorySourceFlags outputFlags) override;

    IBackendInternal::IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const override;
    IBackendInternal::IBackendProfilingContextPtr CreateBackendProfilingContext(
        const IRuntime::CreationOptions&, IBackendProfilingPtr& backendProfiling) override;

    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override;
    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport(const ModelOptions& modelOptions) const override;

    OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph,
                                           const ModelOptions& modelOptions) const override;

    IBackendInternal::IBackendSpecificModelContextPtr CreateBackendSpecificModelContext(
        const ModelOptions& modelOptions) const override;

    std::unique_ptr<ICustomAllocator> GetDefaultAllocator() const override;

    BackendCapabilities GetCapabilities() const override
    {
        return gpuAccCapabilities;
    };

    virtual bool UseCustomMemoryAllocator(std::shared_ptr<ICustomAllocator> allocator,
                                          armnn::Optional<std::string&> errMsg) override
    {
        IgnoreUnused(errMsg);
        ARMNN_LOG(info) << "Using Custom Allocator for ClBackend";

        // Set flag to signal the backend to use a custom memory allocator
        m_CustomAllocator = std::make_shared<ClBackendCustomAllocatorWrapper>(std::move(allocator));
        m_UsingCustomAllocator = true;
        return m_UsingCustomAllocator;
    }

    virtual unsigned int GetNumberOfCacheFiles() const override { return 1; }

    // Cl requires a arm_compute::IAllocator we wrap the Arm NN ICustomAllocator to achieve this
    class ClBackendCustomAllocatorWrapper : public arm_compute::IAllocator
    {
    public:
        ClBackendCustomAllocatorWrapper(std::shared_ptr<ICustomAllocator> alloc) : m_CustomAllocator(alloc)
        {}
        // Inherited methods overridden:
        void* allocate(size_t size, size_t alignment) override
        {
            auto alloc = m_CustomAllocator->allocate(size, alignment);
            return MapAllocatedMemory(alloc, size, m_CustomAllocator->GetMemorySourceType());
        }
        void free(void* ptr) override
        {
            auto hostMemPtr = m_AllocatedBufferMappings[ptr];
            clReleaseMemObject(static_cast<cl_mem>(ptr));
            m_CustomAllocator->free(hostMemPtr);
        }
        std::unique_ptr<arm_compute::IMemoryRegion> make_region(size_t size, size_t alignment) override
        {
            auto hostMemPtr = m_CustomAllocator->allocate(size, alignment);
            cl_mem buffer = MapAllocatedMemory(hostMemPtr, size, m_CustomAllocator->GetMemorySourceType());

            return std::make_unique<ClBackendCustomAllocatorMemoryRegion>(cl::Buffer(buffer),
                                                                          hostMemPtr,
                                                                          m_CustomAllocator->GetMemorySourceType());
        }
    private:
        cl_mem MapAllocatedMemory(void* memory, size_t size, MemorySource source)
        {
            // Round the size of the buffer to a multiple of the CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE
            auto cachelineAlignment =
                    arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
            auto roundedSize = cachelineAlignment + size - (size % cachelineAlignment);

            if (source == MemorySource::Malloc)
            {
                const cl_import_properties_arm importProperties[] =
                        {
                            CL_IMPORT_TYPE_ARM,
                            CL_IMPORT_TYPE_HOST_ARM,
                            0
                        };
                cl_int error = CL_SUCCESS;
                cl_mem buffer = clImportMemoryARM(arm_compute::CLKernelLibrary::get().context().get(),
                                                  CL_MEM_READ_WRITE,
                                                  importProperties,
                                                  memory,
                                                  roundedSize,
                                                  &error);
                if (error == CL_SUCCESS)
                {
                    m_AllocatedBufferMappings.insert(std::make_pair(static_cast<void *>(buffer), memory));
                    return buffer;
                }
                throw armnn::Exception(
                    "Mapping allocated memory from CustomMemoryAllocator failed, errcode: " + std::to_string(error));
            }
            else if (source == MemorySource::DmaBuf)
            {
                const cl_import_properties_arm importProperties[] =
                        {
                            CL_IMPORT_TYPE_ARM,
                            CL_IMPORT_TYPE_DMA_BUF_ARM,
                            CL_IMPORT_DMA_BUF_DATA_CONSISTENCY_WITH_HOST_ARM,
                            CL_TRUE,
                            0
                        };
                cl_int error = CL_SUCCESS;
                cl_mem buffer = clImportMemoryARM(arm_compute::CLKernelLibrary::get().context().get(),
                                                  CL_MEM_READ_WRITE,
                                                  importProperties,
                                                  memory,
                                                  roundedSize,
                                                  &error);
                if (error == CL_SUCCESS)
                {
                    m_AllocatedBufferMappings.insert(std::make_pair(static_cast<void *>(buffer), memory));
                    return buffer;
                }
                throw armnn::Exception(
                        "Mapping allocated memory from CustomMemoryAllocator failed, errcode: "
                         + std::to_string(error));
            }
            else if (source == MemorySource::DmaBufProtected)
            {
                const cl_import_properties_arm importProperties[] =
                        {
                                CL_IMPORT_TYPE_ARM,
                                CL_IMPORT_TYPE_DMA_BUF_ARM,
                                CL_IMPORT_TYPE_PROTECTED_ARM,
                                CL_TRUE,
                                0
                        };
                cl_int error = CL_SUCCESS;
                cl_mem buffer = clImportMemoryARM(arm_compute::CLKernelLibrary::get().context().get(),
                                                  CL_MEM_READ_WRITE,
                                                  importProperties,
                                                  memory,
                                                  roundedSize,
                                                  &error);
                if (error == CL_SUCCESS)
                {
                    m_AllocatedBufferMappings.insert(std::make_pair(static_cast<void *>(buffer), memory));
                    return buffer;
                }
                throw armnn::Exception(
                        "Mapping allocated memory from CustomMemoryAllocator failed, errcode: "
                         + std::to_string(error));
            }
            throw armnn::Exception(
                    "Attempting to allocate memory with unsupported MemorySource type in CustomAllocator");
        }
        std::shared_ptr<ICustomAllocator> m_CustomAllocator;
        std::map<void*, void*> m_AllocatedBufferMappings;
    };

    class ClBackendCustomAllocatorMemoryRegion : public arm_compute::ICLMemoryRegion
    {
    public:
        // We need to have a new version of ICLMemoryRegion which holds a hostMemPtr to allow for cpu copy access
        ClBackendCustomAllocatorMemoryRegion(const cl::Buffer &buffer, void* hostMemPtr, armnn::MemorySource source)
            : ICLMemoryRegion(buffer.getInfo<CL_MEM_SIZE>())
        {
            _mem = buffer;
            m_HostMemPtr = hostMemPtr;
            m_MemorySource = source;
        }

        // Inherited methods overridden :
        void* ptr() override
        {
            return nullptr;
        }

        void* map(cl::CommandQueue &q, bool blocking) override
        {
            armnn::IgnoreUnused(q, blocking);
            if (m_HostMemPtr == nullptr)
            {
                throw armnn::Exception("ClBackend: Attempting to map memory with an invalid host ptr");
            }
            if (_mapping != nullptr)
            {
                throw armnn::Exception("ClBackend: Attempting to map memory which has not yet been unmapped");
            }
            switch (m_MemorySource)
            {
                case armnn::MemorySource::Malloc:
                    _mapping = m_HostMemPtr;
                    return _mapping;
                    break;
                case armnn::MemorySource::DmaBuf:
                case armnn::MemorySource::DmaBufProtected:
                    // If the source is a Dmabuf then the memory ptr should be pointing to an integer value for the fd
                    _mapping = mmap(NULL, _size, PROT_WRITE, MAP_SHARED, *(reinterpret_cast<int*>(m_HostMemPtr)), 0);
                    return _mapping;
                    break;
                default:
                    throw armnn::Exception("ClBackend: Attempting to map imported memory without a valid source");
                    break;
            }
        }

        void unmap(cl::CommandQueue &q) override
        {
            armnn::IgnoreUnused(q);
            switch (m_MemorySource)
            {
                case armnn::MemorySource::Malloc:
                    _mapping = nullptr;
                    break;
                case armnn::MemorySource::DmaBuf:
                case armnn::MemorySource::DmaBufProtected:
                    munmap(_mapping, _size);
                    _mapping = nullptr;
                    break;
                default:
                    throw armnn::Exception("ClBackend: Attempting to unmap imported memory without a valid source");
                    break;
            }
        }
    private:
        void* m_HostMemPtr = nullptr;
        armnn::MemorySource m_MemorySource;
    };

    std::shared_ptr<ClBackendCustomAllocatorWrapper> m_CustomAllocator;
    bool m_UsingCustomAllocator = false;
};

} // namespace armnn
