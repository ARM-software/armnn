//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendRegistry.hpp>
#include <armnn/backends/MemCopyWorkload.hpp>
#include <armnnTestUtils/MockBackend.hpp>
#include <armnnTestUtils/MockTensorHandle.hpp>
#include <backendsCommon/DefaultAllocator.hpp>
#include <backendsCommon/test/MockBackendId.hpp>
#include <SubgraphViewSelector.hpp>

#include "Layer.hpp"

namespace armnn
{

const BackendId& MockBackend::GetIdStatic()
{
    static const BackendId s_Id{MockBackendId()};
    return s_Id;
}

namespace
{
static const BackendId s_Id{ MockBackendId() };
}

MockWorkloadFactory::MockWorkloadFactory(const std::shared_ptr<MockMemoryManager>& memoryManager)
    : m_MemoryManager(memoryManager)
{}

MockWorkloadFactory::MockWorkloadFactory()
    : m_MemoryManager(new MockMemoryManager())
{}

const BackendId& MockWorkloadFactory::GetBackendId() const
{
    return s_Id;
}

std::unique_ptr<IWorkload> MockWorkloadFactory::CreateWorkload(LayerType type,
                                                               const QueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    switch (type)
    {
        case LayerType::MemCopy: {
            auto memCopyQueueDescriptor = PolymorphicDowncast<const MemCopyQueueDescriptor*>(&descriptor);
            if (descriptor.m_Inputs.empty())
            {
                throw InvalidArgumentException("MockWorkloadFactory: CreateMemCopy() expected an input tensor.");
            }
            return std::make_unique<CopyMemGenericWorkload>(*memCopyQueueDescriptor, info);
        }
        default:
            return nullptr;
    }
}

bool IsLayerSupported(const armnn::Layer* layer)
{
    ARMNN_ASSERT(layer != nullptr);

    armnn::LayerType layerType = layer->GetType();
    switch (layerType)
    {
        case armnn::LayerType::Input:
        case armnn::LayerType::Output:
        case armnn::LayerType::Constant:
        case armnn::LayerType::Addition:
        case armnn::LayerType::Convolution2d:
        case armnn::LayerType::ElementwiseBinary:
            // Layer supported
            return true;
        default:
            // Layer unsupported
            return false;
    }
}

bool IsLayerSupported(const armnn::Layer& layer)
{
    return IsLayerSupported(&layer);
}

bool IsLayerOptimizable(const armnn::Layer* layer)
{
    ARMNN_ASSERT(layer != nullptr);

    // A Layer is not optimizable if its name contains "unoptimizable"
    const std::string layerName(layer->GetName());
    bool optimizable = layerName.find("unoptimizable") == std::string::npos;

    return optimizable;
}

bool IsLayerOptimizable(const armnn::Layer& layer)
{
    return IsLayerOptimizable(&layer);
}

} // Anonymous namespace

namespace armnn
{

MockBackendInitialiser::MockBackendInitialiser()
{
    BackendRegistryInstance().Register(MockBackend::GetIdStatic(),
                                       []()
                                       {
                                           return IBackendInternalUniquePtr(new MockBackend);
                                       });
}

MockBackendInitialiser::~MockBackendInitialiser()
{
    try
    {
        BackendRegistryInstance().Deregister(MockBackend::GetIdStatic());
    }
    catch (...)
    {
        std::cerr << "could not deregister mock backend" << std::endl;
    }
}

IBackendInternal::IWorkloadFactoryPtr MockBackend::CreateWorkloadFactory(
    const IBackendInternal::IMemoryManagerSharedPtr& /*memoryManager*/) const
{
    return IWorkloadFactoryPtr{};
}

IBackendInternal::IBackendContextPtr MockBackend::CreateBackendContext(const IRuntime::CreationOptions&) const
{
    return IBackendContextPtr{};
}

IBackendInternal::IBackendProfilingContextPtr MockBackend::CreateBackendProfilingContext(
    const IRuntime::CreationOptions& options, IBackendProfilingPtr& backendProfiling)
{
    IgnoreUnused(options);
    std::shared_ptr<armnn::MockBackendProfilingContext> context =
        std::make_shared<MockBackendProfilingContext>(backendProfiling);
    MockBackendProfilingService::Instance().SetProfilingContextPtr(context);
    return context;
}

IBackendInternal::IMemoryManagerUniquePtr MockBackend::CreateMemoryManager() const
{
    return IMemoryManagerUniquePtr{};
}

IBackendInternal::ILayerSupportSharedPtr MockBackend::GetLayerSupport() const
{
    static ILayerSupportSharedPtr layerSupport{new MockLayerSupport};
    return layerSupport;
}

OptimizationViews MockBackend::OptimizeSubgraphView(const SubgraphView& subgraph) const
{
    // Prepare the optimization views
    OptimizationViews optimizationViews;

    // Get the layers of the input sub-graph
    const SubgraphView::IConnectableLayers& subgraphLayers = subgraph.GetIConnectableLayers();

    // Parse the layers
    SubgraphView::IConnectableLayers supportedLayers;
    SubgraphView::IConnectableLayers unsupportedLayers;
    SubgraphView::IConnectableLayers untouchedLayers;
    std::for_each(subgraphLayers.begin(),
                  subgraphLayers.end(),
                  [&](IConnectableLayer* layer)
                  {
                      bool supported = IsLayerSupported(PolymorphicDowncast<Layer*>(layer));
                      if (supported)
                      {
                          // Layer supported, check if it's optimizable
                          bool optimizable = IsLayerOptimizable(PolymorphicDowncast<Layer*>(layer));
                          if (optimizable)
                          {
                              // Layer fully supported
                              supportedLayers.push_back(layer);
                          }
                          else
                          {
                              // Layer supported but not optimizable
                              untouchedLayers.push_back(layer);
                          }
                      }
                      else
                      {
                          // Layer unsupported
                          unsupportedLayers.push_back(layer);
                      }
                  });

    // Check if there are supported layers
    if (!supportedLayers.empty())
    {
        // Select the layers that are neither inputs or outputs, but that are optimizable
        auto supportedSubgraphSelector = [](const Layer& layer)
        {
            return layer.GetType() != LayerType::Input &&
                layer.GetType() != LayerType::Output &&
                IsLayerSupported(layer) &&
                IsLayerOptimizable(layer);
        };

        // Apply the subgraph selector to the supported layers to group them into sub-graphs were appropriate
        SubgraphView mutableSubgraph(subgraph);
        SubgraphViewSelector::Subgraphs supportedSubgraphs =
                         SubgraphViewSelector::SelectSubgraphs(mutableSubgraph, supportedSubgraphSelector);

        // Create a substitution pair for each supported sub-graph
        std::for_each(supportedSubgraphs.begin(),
                      supportedSubgraphs.end(),
                      [&optimizationViews](const SubgraphView::SubgraphViewPtr& supportedSubgraph)
                      {
                          ARMNN_ASSERT(supportedSubgraph != nullptr);

                          CompiledBlobPtr blobPtr;
                          BackendId backend = MockBackendId();

                          IConnectableLayer* preCompiledLayer =
                                               optimizationViews.GetINetwork()->AddPrecompiledLayer(
                                                   PreCompiledDescriptor(supportedSubgraph->GetNumInputSlots(),
                                                                         supportedSubgraph->GetNumOutputSlots()),
                                                   std::move(blobPtr),
                                                   backend,
                                                   nullptr);

                          SubgraphView substitutionSubgraph(*supportedSubgraph);
                          SubgraphView replacementSubgraph(preCompiledLayer);

                          optimizationViews.AddSubstitution({ substitutionSubgraph, replacementSubgraph });
                      });
    }

    // Check if there are unsupported layers
    if (!unsupportedLayers.empty())
    {
        // Select the layers that are neither inputs or outputs, and are not optimizable
        auto unsupportedSubgraphSelector = [](const Layer& layer)
        {
            return layer.GetType() != LayerType::Input &&
                layer.GetType() != LayerType::Output &&
                !IsLayerSupported(layer);
        };

        // Apply the subgraph selector to the unsupported layers to group them into sub-graphs were appropriate
        SubgraphView mutableSubgraph(subgraph);
        SubgraphViewSelector::Subgraphs unsupportedSubgraphs =
                         SubgraphViewSelector::SelectSubgraphs(mutableSubgraph, unsupportedSubgraphSelector);

        // Add each unsupported sub-graph to the list of failed sub-graphs in the optimizization views
        std::for_each(unsupportedSubgraphs.begin(),
                      unsupportedSubgraphs.end(),
                      [&optimizationViews](const SubgraphView::SubgraphViewPtr& unsupportedSubgraph)
                      {
                          ARMNN_ASSERT(unsupportedSubgraph != nullptr);

                          optimizationViews.AddFailedSubgraph(SubgraphView(*unsupportedSubgraph));
                      });
    }

    // Check if there are untouched layers
    if (!untouchedLayers.empty())
    {
        // Select the layers that are neither inputs or outputs, that are supported but that and are not optimizable
        auto untouchedSubgraphSelector = [](const Layer& layer)
        {
            return layer.GetType() != LayerType::Input &&
                layer.GetType() != LayerType::Output &&
                IsLayerSupported(layer) &&
                !IsLayerOptimizable(layer);
        };

        // Apply the subgraph selector to the untouched layers to group them into sub-graphs were appropriate
        SubgraphView mutableSubgraph(subgraph);
        SubgraphViewSelector::Subgraphs untouchedSubgraphs =
                         SubgraphViewSelector::SelectSubgraphs(mutableSubgraph, untouchedSubgraphSelector);

        // Add each untouched sub-graph to the list of untouched sub-graphs in the optimizization views
        std::for_each(untouchedSubgraphs.begin(),
                      untouchedSubgraphs.end(),
                      [&optimizationViews](const SubgraphView::SubgraphViewPtr& untouchedSubgraph)
                      {
                          ARMNN_ASSERT(untouchedSubgraph != nullptr);

                          optimizationViews.AddUntouchedSubgraph(SubgraphView(*untouchedSubgraph));
                      });
    }

    return optimizationViews;
}

std::unique_ptr<ICustomAllocator> MockBackend::GetDefaultAllocator() const
{
    return std::make_unique<DefaultAllocator>();
}

}    // namespace armnn
