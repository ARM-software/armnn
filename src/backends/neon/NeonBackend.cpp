//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonBackend.hpp"
#include "NeonBackendId.hpp"
#include "NeonWorkloadFactory.hpp"
#include "NeonLayerSupport.hpp"
#include "NeonTensorHandleFactory.hpp"

#include <aclCommon/BaseMemoryManager.hpp>

#include <backendsCommon/BackendRegistry.hpp>
#include <backendsCommon/IBackendContext.hpp>
#include <backendsCommon/IMemoryManager.hpp>

#include <Optimizer.hpp>

#include <arm_compute/runtime/Allocator.h>

#include <boost/cast.hpp>
#include <boost/polymorphic_pointer_cast.hpp>

namespace armnn
{

const BackendId& NeonBackend::GetIdStatic()
{
    static const BackendId s_Id{NeonBackendId()};
    return s_Id;
}

IBackendInternal::IMemoryManagerUniquePtr NeonBackend::CreateMemoryManager() const
{
    return std::make_unique<NeonMemoryManager>(std::make_unique<arm_compute::Allocator>(),
                                               BaseMemoryManager::MemoryAffinity::Offset);
}

IBackendInternal::IWorkloadFactoryPtr NeonBackend::CreateWorkloadFactory(
    const IBackendInternal::IMemoryManagerSharedPtr& memoryManager) const
{
    return std::make_unique<NeonWorkloadFactory>(
        boost::polymorphic_pointer_downcast<NeonMemoryManager>(memoryManager));
}

IBackendInternal::IWorkloadFactoryPtr NeonBackend::CreateWorkloadFactory(
    class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry) const
{
    auto memoryManager = std::make_shared<NeonMemoryManager>(std::make_unique<arm_compute::Allocator>(),
                                                             BaseMemoryManager::MemoryAffinity::Offset);

    tensorHandleFactoryRegistry.RegisterMemoryManager(memoryManager);
    return std::make_unique<NeonWorkloadFactory>(
        boost::polymorphic_pointer_downcast<NeonMemoryManager>(memoryManager));
}

IBackendInternal::IBackendContextPtr NeonBackend::CreateBackendContext(const IRuntime::CreationOptions&) const
{
    return IBackendContextPtr{};
}

IBackendInternal::Optimizations NeonBackend::GetOptimizations() const
{
    return Optimizations{};
}

IBackendInternal::ILayerSupportSharedPtr NeonBackend::GetLayerSupport() const
{
    static ILayerSupportSharedPtr layerSupport{new NeonLayerSupport};
    return layerSupport;
}

OptimizationViews NeonBackend::OptimizeSubgraphView(const SubgraphView& subgraph) const
{
    OptimizationViews optimizationViews;

    optimizationViews.AddUntouchedSubgraph(SubgraphView(subgraph));

    return optimizationViews;
}

std::vector<ITensorHandleFactory::FactoryId> NeonBackend::GetHandleFactoryPreferences() const
{
    return std::vector<ITensorHandleFactory::FactoryId>() = {NeonTensorHandleFactory::GetIdStatic()};
}

void NeonBackend::RegisterTensorHandleFactories(class TensorHandleFactoryRegistry& registry)
{
    auto memoryManager = std::make_shared<NeonMemoryManager>(std::make_unique<arm_compute::Allocator>(),
                                                             BaseMemoryManager::MemoryAffinity::Offset);

    registry.RegisterMemoryManager(memoryManager);
    registry.RegisterFactory(std::make_unique<NeonTensorHandleFactory>(memoryManager));
}

} // namespace armnn
