//
// Copyright © 2022-2025 Arm Ltd and Contributors. All rights reserved.
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

static void LayerToTosaMappings(const Layer& base,
                                std::vector<std::string>& graphInputs,
                                std::vector<std::string>& graphOutputs,
                                std::vector<TosaSerializationOperator*>& operators,
                                std::vector<TosaSerializationTensor*>& tensors)
{
    tosa::TosaSerializationBasicBlock* mappings = GetTosaMappingFromLayer(&base);

    // Loop through inputs to see if there are any graph inputs, if so save them.
    // If it's an input to the graph "input" can be found in the string.
    for (const std::string& blockInputName : mappings->GetInputs())
    {
        if (blockInputName.find("input") != std::string::npos)
        {
            graphInputs.push_back(blockInputName);
        }
    }

    // Loop through outputs to see if there are any graph outputs, if so save them.
    // If it's an output to the graph "output" can be found in the string.
    for (const std::string& blockOutputName : mappings->GetOutputs())
    {
        if (blockOutputName.find("output") != std::string::npos)
        {
            graphOutputs.push_back(blockOutputName);
        }
    }

    auto blockOperators = mappings->GetOperators();
    operators.insert(operators.end(), blockOperators.begin(), blockOperators.end());

    auto blockTensors = mappings->GetTensors();
    tensors.insert(tensors.end(), blockTensors.begin(), blockTensors.end());
}

OptimizationViews TosaRefBackend::OptimizeSubgraphView(const SubgraphView& subgraph,
                                                       const ModelOptions& modelOptions) const
{
    OptimizationViews optimizationViews(modelOptions);

    auto handler = std::make_unique<TosaSerializationHandler>();

    std::vector<std::string> graphInputs;
    std::vector<std::string> graphOutputs;

    std::vector<TosaSerializationOperator*> operators;
    std::vector<TosaSerializationTensor*> tensors;

    bool graphContainsBroadcastReshape = false;

    auto it = subgraph.begin();
    while (it != subgraph.end())
    {
        Layer& base = *(PolymorphicDowncast<Layer*>(*it));

        if (base.GetType() == LayerType::Input ||
            base.GetType() == LayerType::Output)
        {
            ++it;
            continue;
        }

        bool baseContainsBroadcastReshape = false;
        if (base.GetType() == LayerType::ElementwiseBinary)
        {
            auto nextIt = it;
            if (++nextIt != subgraph.end())
            {
                Layer& nextLayer = *(PolymorphicDowncast<Layer*>(*(nextIt)));

                for (uint32_t i = 0; i < base.GetNumInputSlots(); i++)
                {
                    const auto& parentLayer = base.GetInputSlot(i).GetConnectedOutputSlot()->GetOwningLayer();
                    if (parentLayer.GetType() == LayerType::Reshape && &parentLayer == &nextLayer)
                    {
                        LayerToTosaMappings(nextLayer, graphInputs, graphOutputs, operators, tensors);
                        graphContainsBroadcastReshape = true;
                        baseContainsBroadcastReshape = true;
                    }
                }
            }
        }

        LayerToTosaMappings(base, graphInputs, graphOutputs, operators, tensors);

        ++it;
        if (baseContainsBroadcastReshape)
        {
            ++it;
        }
    }

    if (graphContainsBroadcastReshape)
    {
        std::sort(graphInputs.begin(),graphInputs.end());
    }

    // Add all mappings to main block.
    auto* block = new TosaSerializationBasicBlock("main", "main", operators, tensors, graphInputs, graphOutputs);

    std::vector<TosaSerializationBasicBlock*> blocks;
    blocks.emplace_back(block);

    // Add blocks to the main region.
    auto* region = new TosaSerializationRegion("main", blocks);
    handler->GetRegions().emplace_back(region);

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

BackendCapabilities TosaRefBackend::GetCapabilities() const
{
    return BackendCapabilities ("TosaRef",
                                {
                                        {"NonConstWeights", true},
                                        {"ConstantTensorsAsInputs", true}
                                });
}

} // namespace armnn
