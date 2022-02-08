//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonBackend.hpp"
#include "NeonBackendId.hpp"
#include "NeonBackendModelContext.hpp"
#include "NeonWorkloadFactory.hpp"
#include "NeonLayerSupport.hpp"
#include "NeonTensorHandleFactory.hpp"

#include <armnn/BackendRegistry.hpp>
#include <armnn/Descriptors.hpp>

#include <aclCommon/ArmComputeSubgraphUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/BaseMemoryManager.hpp>

#include <armnn/backends/IBackendContext.hpp>
#include <armnn/backends/IMemoryManager.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <neon/workloads/NeonAdditionWorkload.hpp>
#include <neon/workloads/NeonBatchNormalizationWorkload.hpp>
#include <neon/workloads/NeonConvolution2dWorkload.hpp>
#include <neon/workloads/NeonDepthwiseConvolutionWorkload.hpp>
#include <neon/workloads/NeonDivisionWorkload.hpp>
#include <neon/workloads/NeonFullyConnectedWorkload.hpp>
#include <neon/workloads/NeonMultiplicationWorkload.hpp>
#include <neon/workloads/NeonReduceWorkload.hpp>
#include <neon/workloads/NeonSubtractionWorkload.hpp>
#include <backendsCommon/DefaultAllocator.hpp>

#include <Optimizer.hpp>

#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/Allocator.h>

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
        PolymorphicPointerDowncast<NeonMemoryManager>(memoryManager));
}

IBackendInternal::IWorkloadFactoryPtr NeonBackend::CreateWorkloadFactory(
    const IBackendInternal::IMemoryManagerSharedPtr& memoryManager, const ModelOptions& modelOptions) const
{
    return std::make_unique<NeonWorkloadFactory>(
        PolymorphicPointerDowncast<NeonMemoryManager>(memoryManager), CreateBackendSpecificModelContext(modelOptions));
}

IBackendInternal::IWorkloadFactoryPtr NeonBackend::CreateWorkloadFactory(
    class TensorHandleFactoryRegistry& tensorHandleFactoryRegistry) const
{
    auto memoryManager = std::make_shared<NeonMemoryManager>(std::make_unique<arm_compute::Allocator>(),
                                                             BaseMemoryManager::MemoryAffinity::Offset);

    tensorHandleFactoryRegistry.RegisterMemoryManager(memoryManager);

    auto factory = std::make_unique<NeonTensorHandleFactory>(memoryManager);
    // Register copy and import factory pair
    tensorHandleFactoryRegistry.RegisterCopyAndImportFactoryPair(factory->GetId(), factory->GetId());
    // Register the factory
    tensorHandleFactoryRegistry.RegisterFactory(std::move(factory));


    return std::make_unique<NeonWorkloadFactory>(
        PolymorphicPointerDowncast<NeonMemoryManager>(memoryManager));
}

IBackendInternal::IWorkloadFactoryPtr NeonBackend::CreateWorkloadFactory(
    TensorHandleFactoryRegistry& tensorHandleFactoryRegistry, const ModelOptions& modelOptions) const
{
    auto memoryManager = std::make_shared<NeonMemoryManager>(std::make_unique<arm_compute::Allocator>(),
                                                             BaseMemoryManager::MemoryAffinity::Offset);

    tensorHandleFactoryRegistry.RegisterMemoryManager(memoryManager);

    auto factory = std::make_unique<NeonTensorHandleFactory>(memoryManager);
    // Register copy and import factory pair
    tensorHandleFactoryRegistry.RegisterCopyAndImportFactoryPair(factory->GetId(), factory->GetId());
    // Register the factory
    tensorHandleFactoryRegistry.RegisterFactory(std::move(factory));

    return std::make_unique<NeonWorkloadFactory>(
        PolymorphicPointerDowncast<NeonMemoryManager>(memoryManager), CreateBackendSpecificModelContext(modelOptions));
}

IBackendInternal::IBackendContextPtr NeonBackend::CreateBackendContext(const IRuntime::CreationOptions&) const
{
    return IBackendContextPtr{};
}

IBackendInternal::IBackendProfilingContextPtr NeonBackend::CreateBackendProfilingContext(
    const IRuntime::CreationOptions&, IBackendProfilingPtr&)
{
    return IBackendProfilingContextPtr{};
}

IBackendInternal::IBackendSpecificModelContextPtr NeonBackend::CreateBackendSpecificModelContext(
    const ModelOptions& modelOptions) const
{
    return IBackendSpecificModelContextPtr{new NeonBackendModelContext{modelOptions}};
}

IBackendInternal::ILayerSupportSharedPtr NeonBackend::GetLayerSupport() const
{
    static ILayerSupportSharedPtr layerSupport
        {
            new NeonLayerSupport(IBackendInternal::IBackendSpecificModelContextPtr{})
        };
    return layerSupport;
}

IBackendInternal::ILayerSupportSharedPtr NeonBackend::GetLayerSupport(const ModelOptions& modelOptions) const
{
    static ILayerSupportSharedPtr layerSupport
        {
            new NeonLayerSupport(CreateBackendSpecificModelContext(modelOptions))
        };
    return layerSupport;
}

OptimizationViews NeonBackend::OptimizeSubgraphView(const SubgraphView& subgraph) const
{
    OptimizationViews optimizationViews;

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

        // Fuse activation into previous layer if supported by backend
        if ((base.GetType() == LayerType::DepthwiseConvolution2d || base.GetType() == LayerType::Convolution2d
             || base.GetType() == LayerType::BatchNormalization || base.GetType() == LayerType::FullyConnected
             || base.GetType() == LayerType::Addition || base.GetType() == LayerType::Multiplication
             || base.GetType() == LayerType::Subtraction || base.GetType() == LayerType::Division)
            && (base.GetAdditionalInformation<ActivationDescriptor>() == nullptr))
        {
            for (auto output = base.BeginOutputSlots(); output != base.EndOutputSlots(); ++output)
            {
                if (output->GetNumConnections() == 1)
                {
                    for (auto&& childInput : output->GetConnections())
                    {
                        if ((childInput->GetOwningLayer().GetType() == LayerType::Activation) &&
                            (checkDataTypeInputandOutput(childInput->GetOwningLayer())))
                        {
                            Layer& child = childInput->GetOwningLayer();

                            auto* activationLayer = PolymorphicDowncast<ActivationLayer*>(&child);

                            const std::string name = std::string("fused-") + child.GetName() + std::string("-into-") +
                                                     base.GetName();

                            // Get params from activation layer
                            ActivationDescriptor activationDesc = activationLayer->GetParameters();

                            if (base.GetType() == LayerType::Convolution2d)
                            {
                                Convolution2dLayer* baseLayer = PolymorphicDowncast<Convolution2dLayer*>(&base);

                                Optional<TensorInfo> biases;

                                if (baseLayer->GetParameters().m_BiasEnabled)
                                {
                                    biases = baseLayer->m_Bias->GetTensorInfo();
                                }

                                arm_compute::Status status = NeonConvolution2dWorkloadValidate(
                                        baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        baseLayer->GetParameters(),
                                        baseLayer->m_Weight->GetTensorInfo(),
                                        biases,
                                        false,
                                        &activationDesc);

                                if (status)
                                {
                                    FuseConvolution2dLayer<Convolution2dLayer>(optimizationViews,
                                                                               baseLayer,
                                                                               activationLayer,
                                                                               activationDesc,
                                                                               name);
                                    untouched.erase(baseLayer->GetGuid());
                                    untouched.erase(activationLayer->GetGuid());
                                }
                            }
                            else if (base.GetType() == LayerType::DepthwiseConvolution2d)
                            {
                                DepthwiseConvolution2dLayer* baseLayer =
                                        PolymorphicDowncast<DepthwiseConvolution2dLayer*>(&base);

                                Optional<TensorInfo> biases;

                                if (baseLayer->GetParameters().m_BiasEnabled)
                                {
                                    biases = baseLayer->m_Bias->GetTensorInfo();
                                }

                                arm_compute::Status status = NeonDepthwiseConvolutionWorkloadValidate(
                                        baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        baseLayer->GetParameters(),
                                        baseLayer->m_Weight->GetTensorInfo(),
                                        biases,
                                        &activationDesc);

                                if (status)
                                {
                                    FuseDepthwiseConvolution2dLayer<DepthwiseConvolution2dLayer>(optimizationViews,
                                                                                                 baseLayer,
                                                                                                 activationLayer,
                                                                                                 activationDesc,
                                                                                                 name);
                                    untouched.erase(baseLayer->GetGuid());
                                    untouched.erase(activationLayer->GetGuid());
                                }
                            }
                            else if (base.GetType() == LayerType::FullyConnected)
                            {
                                FullyConnectedLayer* baseLayer = PolymorphicDowncast<FullyConnectedLayer*>(&base);
                                Optional<TensorInfo> biases;

                                if (baseLayer->GetParameters().m_BiasEnabled)
                                {
                                    biases = baseLayer->m_Bias->GetTensorInfo();
                                }

                                arm_compute::Status status = NeonFullyConnectedWorkloadValidate(
                                        baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        baseLayer->m_Weight->GetTensorInfo(),
                                        biases,
                                        baseLayer->GetParameters(),
                                        &activationDesc);

                                if (status)
                                {
                                    FuseFullyConnectedLayer<FullyConnectedLayer>(optimizationViews,
                                                                                 baseLayer,
                                                                                 activationLayer,
                                                                                 activationDesc,
                                                                                 name);
                                    untouched.erase(baseLayer->GetGuid());
                                    untouched.erase(activationLayer->GetGuid());
                                }
                            }
                            else if (base.GetType() == LayerType::BatchNormalization)
                            {
                                BatchNormalizationLayer* baseLayer =
                                        PolymorphicDowncast<BatchNormalizationLayer*>(&base);

                                arm_compute::Status status = NeonBatchNormalizationValidate(
                                        baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        baseLayer->m_Mean->GetTensorInfo(),
                                        baseLayer->m_Variance->GetTensorInfo(),
                                        baseLayer->m_Beta->GetTensorInfo(),
                                        baseLayer->m_Gamma->GetTensorInfo(),
                                        baseLayer->GetParameters(),
                                        &activationDesc);

                                if (status)
                                {
                                    BatchNormalizationLayer* replacementLayer =
                                        FuseBatchNormalizationLayer<BatchNormalizationLayer>(optimizationViews,
                                                                                             baseLayer,
                                                                                             activationLayer,
                                                                                             activationDesc,
                                                                                             name);

                                    replacementLayer->m_Beta     = std::move(baseLayer->m_Beta);
                                    replacementLayer->m_Gamma    = std::move(baseLayer->m_Gamma);
                                    replacementLayer->m_Mean     = std::move(baseLayer->m_Mean);
                                    replacementLayer->m_Variance = std::move(baseLayer->m_Variance);
                                    untouched.erase(baseLayer->GetGuid());
                                    untouched.erase(activationLayer->GetGuid());
                                }
                            }
                            else if (base.GetType() == LayerType::Addition)
                            {
                                AdditionLayer* baseLayer = PolymorphicDowncast<AdditionLayer*>(&base);

                                arm_compute::Status status = NeonAdditionWorkloadValidate(
                                        baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        baseLayer->GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo(),
                                        activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        &activationDesc);

                                if (status)
                                {
                                    FuseAdditionLayer<AdditionLayer>(optimizationViews,
                                                                     baseLayer,
                                                                     activationLayer,
                                                                     activationDesc,
                                                                     name);
                                    untouched.erase(baseLayer->GetGuid());
                                    untouched.erase(activationLayer->GetGuid());
                                }
                            }
                            else if (base.GetType() == LayerType::Division)
                            {
                                DivisionLayer* baseLayer = PolymorphicDowncast<DivisionLayer*>(&base);

                                arm_compute::Status status = NeonDivisionWorkloadValidate(
                                        baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        baseLayer->GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo(),
                                        activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        &activationDesc);

                                if (status)
                                {
                                    FuseDivisionLayer<DivisionLayer>(optimizationViews,
                                                                     baseLayer,
                                                                     activationLayer,
                                                                     activationDesc,
                                                                     name);
                                    untouched.erase(baseLayer->GetGuid());
                                    untouched.erase(activationLayer->GetGuid());
                                }
                            }
                            else if (base.GetType() == LayerType::Multiplication)
                            {
                                MultiplicationLayer* baseLayer = PolymorphicDowncast<MultiplicationLayer*>(&base);

                                arm_compute::Status status = NeonMultiplicationWorkloadValidate(
                                        baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        baseLayer->GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo(),
                                        activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        &activationDesc);

                                if (status)
                                {
                                    FuseMultiplicationLayer<MultiplicationLayer>(optimizationViews,
                                                                                 baseLayer,
                                                                                 activationLayer,
                                                                                 activationDesc,
                                                                                 name);
                                    untouched.erase(baseLayer->GetGuid());
                                    untouched.erase(activationLayer->GetGuid());
                                }
                            }
                            else if (base.GetType() == LayerType::Subtraction)
                            {
                                SubtractionLayer* baseLayer = PolymorphicDowncast<SubtractionLayer*>(&base);

                                arm_compute::Status status = NeonSubtractionWorkloadValidate(
                                        baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        baseLayer->GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo(),
                                        activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        &activationDesc);

                                if (status)
                                {
                                    FuseSubtractionLayer<SubtractionLayer>(optimizationViews,
                                                                           baseLayer,
                                                                           activationLayer,
                                                                           activationDesc,
                                                                           name);
                                    untouched.erase(baseLayer->GetGuid());
                                    untouched.erase(activationLayer->GetGuid());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Separate reduce layer with multiple axes into multiple reduce layers with 1 axis.
        if (base.GetType() == LayerType::Reduce)
        {
            ReduceLayer* baseLayer            = PolymorphicDowncast<ReduceLayer*>(&base);
            ReduceDescriptor reduceDescriptor = baseLayer->GetParameters();

            if (!reduceDescriptor.m_vAxis.empty() && reduceDescriptor.m_vAxis.size() > 1)
            {
                // Add new layers to the graph and connect them.
                std::vector<IConnectableLayer*> layers = ChainReduceLayers<ReduceLayer>(optimizationViews,
                                                                                        baseLayer,
                                                                                        reduceDescriptor);

                // Replace existing baselayer with new subgraph.
                ReplaceLayers<ReduceLayer>(optimizationViews, baseLayer, layers);
                untouched.erase(baseLayer->GetGuid());
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

std::vector<ITensorHandleFactory::FactoryId> NeonBackend::GetHandleFactoryPreferences() const
{
    return std::vector<ITensorHandleFactory::FactoryId>() = { NeonTensorHandleFactory::GetIdStatic() };
}

void NeonBackend::RegisterTensorHandleFactories(class TensorHandleFactoryRegistry& registry)
{
    auto memoryManager = std::make_shared<NeonMemoryManager>(std::make_unique<arm_compute::Allocator>(),
                                                             BaseMemoryManager::MemoryAffinity::Offset);

    registry.RegisterMemoryManager(memoryManager);

    auto factory = std::make_unique<NeonTensorHandleFactory>(memoryManager);
    // Register copy and import factory pair
    registry.RegisterCopyAndImportFactoryPair(factory->GetId(), factory->GetId());
    // Register the factory
    registry.RegisterFactory(std::move(factory));
}

std::unique_ptr<ICustomAllocator> NeonBackend::GetDefaultAllocator() const
{
    return std::make_unique<DefaultAllocator>();
}


} // namespace armnn
