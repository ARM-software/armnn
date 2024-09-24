//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IBackendInternal.hpp>
#include <aclCommon/BaseMemoryManager.hpp>

#include <arm_compute/runtime/CL/CLBufferAllocator.h>
#include <arm_compute/runtime/CL/CLMemoryRegion.h>
#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <CL/cl_ext.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadContext.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h>

// System includes for mapping and unmapping memory
#include <sys/mman.h>

namespace armnn
{

/**
 * A structure which contains all the elements needed to execute a fused workload in the GpuFsa Backend
 *
 * @param[in, out]  sketch              A unique pointer to the sketch containing the operators which have been fused.
 * @param[in, out]  TensorInfos         A shared pointer to a GpuWorkloadContext which creates + stores TensorInfos
 * @param[in, out]  inputTensorInfos    A unique pointer to a vector of inputTensorInfos used by the sketch
 * @param[in, out]  outputTensorInfos   A unique pointer to a vector of outputTensorInfos used by the sketch
 *
 */
struct GpuFsaPreCompiledBlob
{
    std::unique_ptr<arm_compute::experimental::dynamic_fusion::GpuWorkloadSketch> sketch = nullptr;
    std::shared_ptr<arm_compute::experimental::dynamic_fusion::GpuWorkloadContext> workloadContext = nullptr;

    std::unique_ptr<std::vector<arm_compute::ITensorInfo*>> inputTensorInfos = nullptr;
    std::unique_ptr<std::vector<arm_compute::ITensorInfo*>> outputTensorInfos = nullptr;
};

// add new capabilities here..
const BackendCapabilities gpuFsaCapabilities("GpuFsa",
                                             {
                                                     {"NonConstWeights", false},
                                                     {"AsyncExecution", false},
                                                     {"ProtectedContentAllocation", false},
                                                     {"ConstantTensorsAsInputs", true},
                                                     {"PreImportIOTensors", false},
                                                     {"ExternallyManagedMemory", false},
                                                     {"MultiAxisPacking", false},
                                                     {"SingleAxisPacking", false},
                                                     {"AllOrNothing", true}
                                             });
ARMNN_NO_DEPRECATE_WARN_BEGIN
class GpuFsaBackend : public IBackendInternal
{
public:
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("The GpuFsa backend will be removed from Arm NN in 24.08", "24.08")
    GpuFsaBackend() : m_CustomAllocator(nullptr) {};
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("The GpuFsa backend will be removed from Arm NN in 24.08", "24.08")
    GpuFsaBackend(std::shared_ptr<ICustomAllocator> allocator)
    {
        UseCustomMemoryAllocator(allocator, armnn::EmptyOptional());
    }
    ~GpuFsaBackend() = default;

    static const BackendId& GetIdStatic();
    const BackendId& GetId() const override { return GetIdStatic(); }

    IBackendInternal::IMemoryManagerUniquePtr CreateMemoryManager() const override;

    IBackendInternal::IWorkloadFactoryPtr CreateWorkloadFactory(
        const IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr) const override;

    IBackendInternal::IWorkloadFactoryPtr CreateWorkloadFactory(TensorHandleFactoryRegistry& registry) const override;

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

    OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph,
                                           const ModelOptions& modelOptions) const override;

    std::unique_ptr<ICustomAllocator> GetDefaultAllocator() const override;

    BackendCapabilities GetCapabilities() const override
    {
        return gpuFsaCapabilities;
    };

    virtual bool UseCustomMemoryAllocator(std::shared_ptr<ICustomAllocator> allocator,
                                          armnn::Optional<std::string&>) override
    {
        ARMNN_LOG(info) << "Using Custom Allocator for GpuFsaBackend";

        // Set flag to signal the backend to use a custom memory allocator
        m_CustomAllocator = std::make_shared<GpuFsaBackendCustomAllocatorWrapper>(std::move(allocator));
        m_UsingCustomAllocator = true;
        return m_UsingCustomAllocator;
    }

    // Cl requires a arm_compute::IAllocator we wrap the Arm NN ICustomAllocator to achieve this
    class GpuFsaBackendCustomAllocatorWrapper : public arm_compute::IAllocator
    {
    public:
        GpuFsaBackendCustomAllocatorWrapper(std::shared_ptr<ICustomAllocator> alloc) : m_CustomAllocator(alloc)
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

    std::shared_ptr<GpuFsaBackendCustomAllocatorWrapper> m_CustomAllocator;
    bool m_UsingCustomAllocator = false;
};
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
