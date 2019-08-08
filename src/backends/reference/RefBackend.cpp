//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefBackend.hpp"
#include "RefBackendId.hpp"
#include "RefWorkloadFactory.hpp"
#include "RefLayerSupport.hpp"

#include <backendsCommon/IBackendContext.hpp>
#include <backendsCommon/IMemoryManager.hpp>
#include <backendsCommon/BackendRegistry.hpp>

#include <Optimizer.hpp>

#include <boost/cast.hpp>
#include <boost/polymorphic_pointer_cast.hpp>

namespace armnn
{

const BackendId& RefBackend::GetIdStatic()
{
    static const BackendId s_Id{RefBackendId()};
    return s_Id;
}

IBackendInternal::IWorkloadFactoryPtr RefBackend::CreateWorkloadFactory(
    const IBackendInternal::IMemoryManagerSharedPtr& memoryManager) const
{
    return std::make_unique<RefWorkloadFactory>(boost::polymorphic_pointer_downcast<RefMemoryManager>(memoryManager));
}

IBackendInternal::IBackendContextPtr RefBackend::CreateBackendContext(const IRuntime::CreationOptions&) const
{
    return IBackendContextPtr{};
}

IBackendInternal::IMemoryManagerUniquePtr RefBackend::CreateMemoryManager() const
{
    return std::make_unique<RefMemoryManager>();
}

IBackendInternal::Optimizations RefBackend::GetOptimizations() const
{
    return Optimizations{};
}

IBackendInternal::ILayerSupportSharedPtr RefBackend::GetLayerSupport() const
{
    static ILayerSupportSharedPtr layerSupport{new RefLayerSupport};
    return layerSupport;
}

OptimizationViews RefBackend::OptimizeSubgraphView(const SubgraphView& subgraph) const
{
    OptimizationViews optimizationViews;

    optimizationViews.AddUntouchedSubgraph(SubgraphView(subgraph));

    return optimizationViews;
}

} // namespace armnn
