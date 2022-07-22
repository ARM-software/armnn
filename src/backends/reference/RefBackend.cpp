//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefBackend.hpp"
#include "RefBackendId.hpp"
#include "RefWorkloadFactory.hpp"
#include "RefLayerSupport.hpp"
#include "RefTensorHandleFactory.hpp"

#include <armnn/BackendRegistry.hpp>
#include <armnn/backends/IBackendContext.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <backendsCommon/DefaultAllocator.hpp>
#include <backendsCommon/SubgraphUtils.hpp>

#include <Optimizer.hpp>

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
    return std::make_unique<RefWorkloadFactory>(PolymorphicPointerDowncast<RefMemoryManager>(memoryManager));
}

IBackendInternal::IWorkloadFactoryPtr RefBackend::CreateWorkloadFactory(
    class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry) const
{
    auto memoryManager = std::make_shared<RefMemoryManager>();

    tensorHandleFactoryRegistry.RegisterMemoryManager(memoryManager);

    std::unique_ptr<RefTensorHandleFactory> factory = std::make_unique<RefTensorHandleFactory>(memoryManager);
    // Register copy and import factory pair
    tensorHandleFactoryRegistry.RegisterCopyAndImportFactoryPair(factory->GetId(), factory->GetId());
    // Register the factory
    tensorHandleFactoryRegistry.RegisterFactory(std::move(factory));

    return std::make_unique<RefWorkloadFactory>(PolymorphicPointerDowncast<RefMemoryManager>(memoryManager));
}

IBackendInternal::IBackendContextPtr RefBackend::CreateBackendContext(const IRuntime::CreationOptions&) const
{
    return IBackendContextPtr{};
}

IBackendInternal::IBackendProfilingContextPtr RefBackend::CreateBackendProfilingContext(
    const IRuntime::CreationOptions&, IBackendProfilingPtr&)
{
    return IBackendProfilingContextPtr{};
}

IBackendInternal::IMemoryManagerUniquePtr RefBackend::CreateMemoryManager() const
{
    return std::make_unique<RefMemoryManager>();
}

IBackendInternal::ILayerSupportSharedPtr RefBackend::GetLayerSupport() const
{
    static ILayerSupportSharedPtr layerSupport{new RefLayerSupport};
    return layerSupport;
}

OptimizationViews RefBackend::OptimizeSubgraphView(const SubgraphView& subgraph,
                                                   const ModelOptions& modelOptions) const
{
    OptimizationViews optimizationViews(modelOptions);

    auto it = subgraph.endIConnectable();
    std::map<LayerGuid, Layer*> untouched;

    while (it != subgraph.beginIConnectable())
    {
        --it;
        Layer& base = *(PolymorphicDowncast<Layer*>(*it));
        untouched.insert({base.GetGuid(), &base});
    }

    it = subgraph.endIConnectable();
    while (it != subgraph.beginIConnectable())
    {
        --it;
        Layer& base = *(PolymorphicDowncast<Layer*>(*it));

        // Special case to fuse padding into average pooling 2d for quantized datatype.
        // Required to be done as a backend specific optimization as Neon does not support this special case.
        if (base.GetType() == LayerType::Pooling2d)
        {
            Pooling2dLayer* baseLayer = PolymorphicDowncast<Pooling2dLayer*>(&base);
            Pooling2dDescriptor poolingDescriptor = baseLayer->GetParameters();

            if (baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer().GetType() == LayerType::Pad)
            {
                PadLayer* padLayer = PolymorphicDowncast<PadLayer*>(
                    &baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer());
                if (padLayer->GetOutputSlot(0).GetNumConnections() == 1 &&
                    optimizations::pad_fold::TryFoldPadIntoLayer2d(padLayer->GetParameters(),
                                                                   poolingDescriptor,
                                                                   padLayer->GetOutputSlot().GetTensorInfo(),
                                                                   true))
                {
                    FoldPadIntoAveragePool2d<Pooling2dLayer>(optimizationViews, baseLayer,
                                                             poolingDescriptor, padLayer);
                    untouched.erase(baseLayer->GetGuid());
                    untouched.erase(padLayer->GetGuid());
                }
            }
        }
    }

    if (optimizationViews.GetSubstitutions().empty())
    {
        optimizationViews.AddUntouchedSubgraph(SubgraphView(subgraph));
    }
    else
    {
        ReportUntouchedLayers(optimizationViews, untouched);
    }

    return optimizationViews;
}

std::vector<ITensorHandleFactory::FactoryId> RefBackend::GetHandleFactoryPreferences() const
{
    return std::vector<ITensorHandleFactory::FactoryId> { RefTensorHandleFactory::GetIdStatic() };
}

void RefBackend::RegisterTensorHandleFactories(class TensorHandleFactoryRegistry& registry)
{
    auto memoryManager = std::make_shared<RefMemoryManager>();

    registry.RegisterMemoryManager(memoryManager);

    std::unique_ptr<RefTensorHandleFactory> factory = std::make_unique<RefTensorHandleFactory>(memoryManager);

    // Register copy and import factory pair
    registry.RegisterCopyAndImportFactoryPair(factory->GetId(), factory->GetId());
    // Register the factory
    registry.RegisterFactory(std::move(factory));
}

std::unique_ptr<ICustomAllocator> RefBackend::GetDefaultAllocator() const
{
    return std::make_unique<DefaultAllocator>();
}

ExecutionData RefBackend::CreateExecutionData(WorkingMemDescriptor& workingMemDescriptor) const
{
    ExecutionData executionData;
    executionData.m_Data = &workingMemDescriptor;
    return executionData;
}

void RefBackend::UpdateExecutionData(ExecutionData& executionData, WorkingMemDescriptor& workingMemDescriptor) const
{
    executionData.m_Data = &workingMemDescriptor;
}

} // namespace armnn
