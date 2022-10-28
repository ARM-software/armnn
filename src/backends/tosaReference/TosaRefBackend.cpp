//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaRefBackend.hpp"
#include "TosaRefBackendId.hpp"
#include "TosaRefWorkloadFactory.hpp"
#include "TosaRefLayerSupport.hpp"
#include "TosaRefTensorHandleFactory.hpp"

#include <tosaCommon/TosaMappings.hpp>
#include <armnn/BackendRegistry.hpp>
#include <armnn/backends/IBackendContext.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <backendsCommon/DefaultAllocator.hpp>
#include <backendsCommon/SubgraphUtils.hpp>

#include <Optimizer.hpp>

namespace armnn
{

// Utility function to construct a valid Deleter for TosaSerializationHandler ptrs passed back to ArmNN
template <typename T>
void DeleteAsType(const void* const blob)
{
    delete static_cast<const T*>(blob);
}

const BackendId& TosaRefBackend::GetIdStatic()
{
    static const BackendId s_Id{TosaRefBackendId()};
    return s_Id;
}

IBackendInternal::IWorkloadFactoryPtr TosaRefBackend::CreateWorkloadFactory(
    const IBackendInternal::IMemoryManagerSharedPtr& memoryManager) const
{
    return std::make_unique<TosaRefWorkloadFactory>(PolymorphicPointerDowncast<TosaRefMemoryManager>(memoryManager));
}

IBackendInternal::IWorkloadFactoryPtr TosaRefBackend::CreateWorkloadFactory(
    class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry) const
{
    auto memoryManager = std::make_shared<TosaRefMemoryManager>();

    tensorHandleFactoryRegistry.RegisterMemoryManager(memoryManager);

    auto factory = std::make_unique<TosaRefTensorHandleFactory>(memoryManager);
    // Register copy and import factory pair
    tensorHandleFactoryRegistry.RegisterCopyAndImportFactoryPair(factory->GetId(), factory->GetId());
    // Register the factory
    tensorHandleFactoryRegistry.RegisterFactory(std::move(factory));

    return std::make_unique<TosaRefWorkloadFactory>(PolymorphicPointerDowncast<TosaRefMemoryManager>(memoryManager));
}

IBackendInternal::IBackendContextPtr TosaRefBackend::CreateBackendContext(const IRuntime::CreationOptions&) const
{
    return IBackendContextPtr{};
}

IBackendInternal::IBackendProfilingContextPtr TosaRefBackend::CreateBackendProfilingContext(
    const IRuntime::CreationOptions&, IBackendProfilingPtr&)
{
    return IBackendProfilingContextPtr{};
}

IBackendInternal::IMemoryManagerUniquePtr TosaRefBackend::CreateMemoryManager() const
{
    return std::make_unique<TosaRefMemoryManager>();
}

IBackendInternal::ILayerSupportSharedPtr TosaRefBackend::GetLayerSupport() const
{
    static ILayerSupportSharedPtr layerSupport{new TosaRefLayerSupport};
    return layerSupport;
}

OptimizationViews TosaRefBackend::OptimizeSubgraphView(const SubgraphView& subgraph,
                                                       const ModelOptions& modelOptions) const
{
    OptimizationViews optimizationViews(modelOptions);
    auto handler = std::make_unique<TosaSerializationHandler>();

    auto it = subgraph.endIConnectable();
    while (it != subgraph.beginIConnectable())
    {
        --it;
        Layer &base = *(PolymorphicDowncast<Layer*>(*it));

        if(base.GetType() == armnn::LayerType::Input ||
           base.GetType() == armnn::LayerType::Output)
        {
            continue;
        }

        tosa::TosaSerializationBasicBlock* mappings = GetTosaMappingFromLayer(&base);
        handler.get()->GetBlocks().push_back(mappings);
    }

    auto compiledBlob =
            std::make_unique<PreCompiledObjectPtr>(handler.release(), DeleteAsType<TosaSerializationHandler>);

    IConnectableLayer* preCompiledLayer = optimizationViews.GetINetwork()->AddPrecompiledLayer(
            PreCompiledDescriptor(subgraph.GetNumInputSlots(), subgraph.GetNumOutputSlots()),
            std::move(*compiledBlob),
            armnn::Optional<BackendId>(GetId()),
            "TOSA_Pre_Compiled_Layer");

    // Copy the output tensor infos from sub-graph
    for (unsigned int i = 0; i < subgraph.GetNumOutputSlots(); i++)
    {
        preCompiledLayer->GetOutputSlot(i).SetTensorInfo(subgraph.GetIOutputSlot(i)->GetTensorInfo());
    }

    optimizationViews.AddSubstitution({ std::move(subgraph), SubgraphView(preCompiledLayer) });
    return optimizationViews;
}


std::vector<ITensorHandleFactory::FactoryId> TosaRefBackend::GetHandleFactoryPreferences() const
{
    return std::vector<ITensorHandleFactory::FactoryId> { TosaRefTensorHandleFactory::GetIdStatic() };
}

void TosaRefBackend::RegisterTensorHandleFactories(class TensorHandleFactoryRegistry& registry)
{
    auto memoryManager = std::make_shared<TosaRefMemoryManager>();

    registry.RegisterMemoryManager(memoryManager);

    auto factory = std::make_unique<TosaRefTensorHandleFactory>(memoryManager);

    // Register copy and import factory pair
    registry.RegisterCopyAndImportFactoryPair(factory->GetId(), factory->GetId());
    // Register the factory
    registry.RegisterFactory(std::move(factory));
}

std::unique_ptr<ICustomAllocator> TosaRefBackend::GetDefaultAllocator() const
{
    return std::make_unique<DefaultAllocator>();
}

} // namespace armnn
