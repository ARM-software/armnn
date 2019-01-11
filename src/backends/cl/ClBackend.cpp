//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBackend.hpp"
#include "ClBackendId.hpp"
#include "ClWorkloadFactory.hpp"
#include "ClBackendContext.hpp"
#include "ClLayerSupport.hpp"

#include <aclCommon/BaseMemoryManager.hpp>

#include <backendsCommon/BackendRegistry.hpp>
#include <backendsCommon/IBackendContext.hpp>
#include <backendsCommon/IMemoryManager.hpp>

#include <Optimizer.hpp>

#include <arm_compute/runtime/CL/CLBufferAllocator.h>

#include <boost/polymorphic_pointer_cast.hpp>

namespace armnn
{

namespace
{

static BackendRegistry::StaticRegistryInitializer g_RegisterHelper
{
    BackendRegistryInstance(),
    ClBackend::GetIdStatic(),
    []()
    {
        return IBackendInternalUniquePtr(new ClBackend);
    }
};

}

const BackendId& ClBackend::GetIdStatic()
{
    static const BackendId s_Id{ClBackendId()};
    return s_Id;
}

IBackendInternal::IMemoryManagerUniquePtr ClBackend::CreateMemoryManager() const
{
    return std::make_unique<ClMemoryManager>(std::make_unique<arm_compute::CLBufferAllocator>());
}

IBackendInternal::IWorkloadFactoryPtr ClBackend::CreateWorkloadFactory(
    const IBackendInternal::IMemoryManagerSharedPtr& memoryManager) const
{
    return std::make_unique<ClWorkloadFactory>(
        boost::polymorphic_pointer_downcast<ClMemoryManager>(memoryManager));
}

IBackendInternal::IBackendContextPtr
ClBackend::CreateBackendContext(const IRuntime::CreationOptions& options) const
{
    return IBackendContextPtr{new ClBackendContext{options}};
}

IBackendInternal::ISubGraphConverterPtr ClBackend::CreateSubGraphConverter(
    const std::shared_ptr<SubGraph>& subGraph) const
{
    return ISubGraphConverterPtr{};
}

IBackendInternal::Optimizations ClBackend::GetOptimizations() const
{
    return Optimizations{};
}

IBackendInternal::ILayerSupportSharedPtr ClBackend::GetLayerSupport() const
{
    static ILayerSupportSharedPtr layerSupport{new ClLayerSupport};
    return layerSupport;
}

} // namespace armnn
