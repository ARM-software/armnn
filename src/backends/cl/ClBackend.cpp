//
// Copyright © 2017-2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBackend.hpp"
#include "ClBackendContext.hpp"
#include "ClBackendDefaultAllocator.hpp"
#include "ClBackendId.hpp"
#include "ClBackendModelContext.hpp"
#include "ClImportTensorHandleFactory.hpp"
#include "ClLayerSupport.hpp"
#include "ClTensorHandleFactory.hpp"
#include "ClWorkloadFactory.hpp"

#include <armnn/BackendRegistry.hpp>
#include <armnn/Descriptors.hpp>

#include <aclCommon/ArmComputeSubgraphUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/BaseMemoryManager.hpp>

#include <armnn/backends/IBackendContext.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include "workloads/ClAdditionWorkload.hpp"
#include "workloads/ClBatchNormalizationFloatWorkload.hpp"
#include "workloads/ClConvolution2dWorkload.hpp"
#include "workloads/ClDepthwiseConvolutionWorkload.hpp"
#include "workloads/ClDivisionWorkload.hpp"
#include "workloads/ClFullyConnectedWorkload.hpp"
#include "workloads/ClMultiplicationWorkload.hpp"
#include "workloads/ClReduceWorkload.hpp"
#include "workloads/ClSubtractionWorkload.hpp"

#include <Optimizer.hpp>

#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/CL/CLBufferAllocator.h>

namespace armnn
{

const BackendId& ClBackend::GetIdStatic()
{
    static const BackendId s_Id{ClBackendId()};
    return s_Id;
}

IBackendInternal::IMemoryManagerUniquePtr ClBackend::CreateMemoryManager() const
{
    if (m_UsingCustomAllocator)
    {
        return std::make_unique<ClMemoryManager>(m_CustomAllocator);
    }
    return std::make_unique<ClMemoryManager>(std::make_unique<arm_compute::CLBufferAllocator>());
}

IBackendInternal::IWorkloadFactoryPtr ClBackend::CreateWorkloadFactory(
    const IBackendInternal::IMemoryManagerSharedPtr& memoryManager) const
{
    return std::make_unique<ClWorkloadFactory>(
        PolymorphicPointerDowncast<ClMemoryManager>(memoryManager));
}

IBackendInternal::IWorkloadFactoryPtr ClBackend::CreateWorkloadFactory(
    const IBackendInternal::IMemoryManagerSharedPtr& memoryManager, const ModelOptions& modelOptions) const
{
    return std::make_unique<ClWorkloadFactory>(
        PolymorphicPointerDowncast<ClMemoryManager>(memoryManager), CreateBackendSpecificModelContext(modelOptions));
}

IBackendInternal::IWorkloadFactoryPtr ClBackend::CreateWorkloadFactory(
    TensorHandleFactoryRegistry& registry) const
{
    std::shared_ptr<ClMemoryManager> memoryManager;
    if (m_UsingCustomAllocator)
    {
        memoryManager = std::make_shared<ClMemoryManager>(m_CustomAllocator);
    }
    else
    {
        memoryManager = std::make_shared<ClMemoryManager>(std::make_unique<arm_compute::CLBufferAllocator>());
    }

    std::unique_ptr<ITensorHandleFactory> factory = std::make_unique<ClTensorHandleFactory>(memoryManager);
    std::unique_ptr<ITensorHandleFactory> importFactory = std::make_unique<ClImportTensorHandleFactory>(
        static_cast<MemorySourceFlags>(MemorySource::Malloc), static_cast<MemorySourceFlags>(MemorySource::Malloc));

    registry.RegisterCopyAndImportFactoryPair(factory->GetId(), importFactory->GetId());
    registry.RegisterCopyAndImportFactoryPair(importFactory->GetId(), factory->GetId());

    registry.RegisterMemoryManager(memoryManager);
    registry.RegisterFactory(std::move(factory));
    registry.RegisterFactory(std::move(importFactory));

    return std::make_unique<ClWorkloadFactory>(
            PolymorphicPointerDowncast<ClMemoryManager>(memoryManager));
}

IBackendInternal::IWorkloadFactoryPtr ClBackend::CreateWorkloadFactory(
    TensorHandleFactoryRegistry& registry, const ModelOptions& modelOptions) const
{
    std::shared_ptr<ClMemoryManager> memoryManager;
    if (m_UsingCustomAllocator)
    {
        memoryManager = std::make_shared<ClMemoryManager>(m_CustomAllocator);
    }
    else
    {
        memoryManager = std::make_shared<ClMemoryManager>(std::make_unique<arm_compute::CLBufferAllocator>());
    }

    std::unique_ptr<ITensorHandleFactory> factory = std::make_unique<ClTensorHandleFactory>(memoryManager);
    std::unique_ptr<ITensorHandleFactory> importFactory = std::make_unique<ClImportTensorHandleFactory>(
        static_cast<MemorySourceFlags>(MemorySource::Malloc), static_cast<MemorySourceFlags>(MemorySource::Malloc));

    registry.RegisterCopyAndImportFactoryPair(factory->GetId(), importFactory->GetId());
    registry.RegisterCopyAndImportFactoryPair(importFactory->GetId(), factory->GetId());

    registry.RegisterMemoryManager(memoryManager);
    registry.RegisterFactory(std::move(factory));
    registry.RegisterFactory(std::move(importFactory));

    return std::make_unique<ClWorkloadFactory>(
        PolymorphicPointerDowncast<ClMemoryManager>(memoryManager), CreateBackendSpecificModelContext(modelOptions));
}

IBackendInternal::IWorkloadFactoryPtr ClBackend::CreateWorkloadFactory(
    TensorHandleFactoryRegistry& registry,
    const ModelOptions& modelOptions,
    MemorySourceFlags inputFlags,
    MemorySourceFlags outputFlags) const
{
    // To allow force import if inputFlags/outputFlags are Undefined, set it as Malloc
    if (inputFlags == static_cast<MemorySourceFlags>(MemorySource::Undefined))
    {
        inputFlags = static_cast<MemorySourceFlags>(MemorySource::Malloc);
    }
    if (outputFlags == static_cast<MemorySourceFlags>(MemorySource::Undefined))
    {
        outputFlags = static_cast<MemorySourceFlags>(MemorySource::Malloc);
    }
    std::shared_ptr<ClMemoryManager> memoryManager;
    if (m_UsingCustomAllocator)
    {
        memoryManager = std::make_shared<ClMemoryManager>(m_CustomAllocator);
    }
    else
    {
        memoryManager = std::make_shared<ClMemoryManager>(std::make_unique<arm_compute::CLBufferAllocator>());
    }

    std::unique_ptr<ITensorHandleFactory> factory = std::make_unique<ClTensorHandleFactory>(memoryManager);
    std::unique_ptr<ITensorHandleFactory> importFactory = std::make_unique<ClImportTensorHandleFactory>(
            inputFlags, outputFlags);

    registry.RegisterCopyAndImportFactoryPair(factory->GetId(), importFactory->GetId());
    registry.RegisterCopyAndImportFactoryPair(importFactory->GetId(), factory->GetId());

    registry.RegisterMemoryManager(memoryManager);
    registry.RegisterFactory(std::move(factory));
    registry.RegisterFactory(std::move(importFactory));

    return std::make_unique<ClWorkloadFactory>(
        PolymorphicPointerDowncast<ClMemoryManager>(memoryManager), CreateBackendSpecificModelContext(modelOptions));
}

std::vector<ITensorHandleFactory::FactoryId> ClBackend::GetHandleFactoryPreferences() const
{
    return std::vector<ITensorHandleFactory::FactoryId> {ClTensorHandleFactory::GetIdStatic(),
                                                         ClImportTensorHandleFactory::GetIdStatic()};
}

void ClBackend::RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry)
{
    std::shared_ptr<ClMemoryManager> memoryManager;
    if (m_UsingCustomAllocator)
    {
        memoryManager = std::make_shared<ClMemoryManager>(m_CustomAllocator);
    }
    else
    {
        memoryManager = std::make_shared<ClMemoryManager>(std::make_unique<arm_compute::CLBufferAllocator>());
    }

    std::unique_ptr<ITensorHandleFactory> factory = std::make_unique<ClTensorHandleFactory>(memoryManager);
    std::unique_ptr<ITensorHandleFactory> importFactory = std::make_unique<ClImportTensorHandleFactory>(
        static_cast<MemorySourceFlags>(MemorySource::Malloc), static_cast<MemorySourceFlags>(MemorySource::Malloc));

    registry.RegisterCopyAndImportFactoryPair(factory->GetId(), importFactory->GetId());
    registry.RegisterCopyAndImportFactoryPair(importFactory->GetId(), factory->GetId());

    registry.RegisterMemoryManager(memoryManager);
    registry.RegisterFactory(std::move(factory));
    registry.RegisterFactory(std::move(importFactory));

}

void ClBackend::RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry,
                                              MemorySourceFlags inputFlags,
                                              MemorySourceFlags outputFlags)
{
    // To allow force import if inputFlags/outputFlags are Undefined, set it as Malloc
    if (inputFlags == static_cast<MemorySourceFlags>(MemorySource::Undefined))
    {
        inputFlags = static_cast<MemorySourceFlags>(MemorySource::Malloc);
    }
    if (outputFlags == static_cast<MemorySourceFlags>(MemorySource::Undefined))
    {
        outputFlags = static_cast<MemorySourceFlags>(MemorySource::Malloc);
    }
    std::shared_ptr<ClMemoryManager> memoryManager;
    if (m_UsingCustomAllocator)
    {
        memoryManager = std::make_shared<ClMemoryManager>(m_CustomAllocator);
    }
    else
    {
        memoryManager = std::make_shared<ClMemoryManager>(std::make_unique<arm_compute::CLBufferAllocator>());
    }

    std::unique_ptr<ITensorHandleFactory> factory = std::make_unique<ClTensorHandleFactory>(memoryManager);
    std::unique_ptr<ITensorHandleFactory> importFactory = std::make_unique<ClImportTensorHandleFactory>(
            inputFlags, outputFlags);

    registry.RegisterCopyAndImportFactoryPair(factory->GetId(), importFactory->GetId());
    registry.RegisterCopyAndImportFactoryPair(importFactory->GetId(), factory->GetId());

    registry.RegisterMemoryManager(memoryManager);
    registry.RegisterFactory(std::move(factory));
    registry.RegisterFactory(std::move(importFactory));
}

IBackendInternal::IBackendContextPtr ClBackend::CreateBackendContext(const IRuntime::CreationOptions& options) const
{
    return IBackendContextPtr{new ClBackendContext{options}};
}

IBackendInternal::IBackendProfilingContextPtr ClBackend::CreateBackendProfilingContext(
    const IRuntime::CreationOptions&, IBackendProfilingPtr&)
{
    return IBackendProfilingContextPtr{};
}

IBackendInternal::IBackendSpecificModelContextPtr ClBackend::CreateBackendSpecificModelContext(
    const ModelOptions& modelOptions) const
{
    return IBackendSpecificModelContextPtr{new ClBackendModelContext{modelOptions}};
}

IBackendInternal::ILayerSupportSharedPtr ClBackend::GetLayerSupport() const
{
    static ILayerSupportSharedPtr layerSupport
        {
            new ClLayerSupport(IBackendInternal::IBackendSpecificModelContextPtr{})
        };
    return layerSupport;
}

IBackendInternal::ILayerSupportSharedPtr ClBackend::GetLayerSupport(const ModelOptions& modelOptions) const
{
    static ILayerSupportSharedPtr layerSupport
    {
        new ClLayerSupport(CreateBackendSpecificModelContext(modelOptions))
    };
    return layerSupport;
}

std::unique_ptr<ICustomAllocator> ClBackend::GetDefaultAllocator() const
{
    return std::make_unique<ClBackendDefaultAllocator>();
}

OptimizationViews ClBackend::OptimizeSubgraphView(const SubgraphView& subgraph,
                                                  const ModelOptions& modelOptions) const
{
    OptimizationViews optimizationViews(modelOptions);

    auto it = subgraph.endIConnectable();
    bool isFastMathEnabled = false;
    std::map<LayerGuid, Layer*> untouched;

    while (it != subgraph.beginIConnectable())
    {
        --it;
        Layer& base = *(PolymorphicDowncast<Layer*>(*it));
        untouched.insert({base.GetGuid(), &base});
    }

    it = subgraph.endIConnectable();
#if defined(ARMCOMPUTECL_ENABLED)
    IBackendInternal::IBackendSpecificModelContextPtr modelContextPtr = CreateBackendSpecificModelContext(modelOptions);

    if (modelContextPtr)
    {
        auto clModelOptions = dynamic_cast<ClBackendModelContext*>(modelContextPtr.get());
        if (clModelOptions)
        {
            isFastMathEnabled = clModelOptions->IsFastMathEnabled();
        }
    }
#endif
    while (it != subgraph.beginIConnectable())
    {
        --it;
        Layer& base = *(PolymorphicDowncast<Layer*>(*it));

        // Fuse activation into previous layer if supported by backend
        if ((base.GetType() == LayerType::DepthwiseConvolution2d || base.GetType() == LayerType::Convolution2d
            || base.GetType() == LayerType::BatchNormalization || base.GetType() == LayerType::FullyConnected
            || base.GetType() == LayerType::Addition || base.GetType() == LayerType::Multiplication
            || base.GetType() == LayerType::Subtraction || base.GetType() == LayerType::Division
            || base.GetType() == LayerType::ElementwiseBinary)
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
                                    biases = baseLayer->GetInputSlot(2).GetConnectedOutputSlot()->GetTensorInfo();
                                }

                                arm_compute::Status status = ClConvolution2dWorkloadValidate(
                                        baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        baseLayer->GetParameters(),
                                        baseLayer->GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo(),
                                        biases,
                                        isFastMathEnabled,
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
                                    biases = baseLayer->GetInputSlot(2).GetConnectedOutputSlot()->GetTensorInfo();
                                }

                                arm_compute::Status status = ClDepthwiseConvolutionWorkloadValidate(
                                        baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        baseLayer->GetParameters(),
                                        baseLayer->GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo(),
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
                                FullyConnectedDescriptor descriptor = baseLayer->GetParameters();

                                // As bias is optional only try to get TensorInfo from input if bias is enabled.
                                Optional<TensorInfo> biases;
                                if (descriptor.m_BiasEnabled)
                                {
                                    biases = baseLayer->GetInputSlot(2).GetConnectedOutputSlot()->GetTensorInfo();
                                }

                                arm_compute::Status status = ClFullyConnectedWorkloadValidate(
                                        baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        baseLayer->GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo(),
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

                                arm_compute::Status status = ClBatchNormalizationValidate(
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

                                arm_compute::Status status = ClAdditionValidate(
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

                                arm_compute::Status status = ClDivisionWorkloadValidate(
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

                                arm_compute::Status status = ClMultiplicationWorkloadValidate(
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

                                arm_compute::Status status = ClSubtractionValidate(
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
                            else if (base.GetType() == LayerType::ElementwiseBinary)
                            {
                                ElementwiseBinaryLayer* baseLayer = PolymorphicDowncast<ElementwiseBinaryLayer*>(&base);

                                if (baseLayer->GetParameters().m_Operation == BinaryOperation::Add)
                                {
                                    arm_compute::Status status = ClAdditionValidate(
                                            baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                            baseLayer->GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo(),
                                            activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                            &activationDesc);

                                    if (status)
                                    {
                                        FuseElementwiseBinaryLayer<ElementwiseBinaryLayer>(optimizationViews,
                                                                                           baseLayer,
                                                                                           activationLayer,
                                                                                           activationDesc,
                                                                                           BinaryOperation::Add,
                                                                                           name);
                                        untouched.erase(baseLayer->GetGuid());
                                        untouched.erase(activationLayer->GetGuid());
                                    }
                                }
                                else if (baseLayer->GetParameters().m_Operation == BinaryOperation::Div)
                                {
                                    arm_compute::Status status = ClDivisionWorkloadValidate(
                                            baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                            baseLayer->GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo(),
                                            activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                            &activationDesc);

                                    if (status)
                                    {
                                        FuseElementwiseBinaryLayer<ElementwiseBinaryLayer>(optimizationViews,
                                                                                           baseLayer,
                                                                                           activationLayer,
                                                                                           activationDesc,
                                                                                           BinaryOperation::Div,
                                                                                           name);
                                        untouched.erase(baseLayer->GetGuid());
                                        untouched.erase(activationLayer->GetGuid());
                                    }
                                }
                                else if (baseLayer->GetParameters().m_Operation == BinaryOperation::Mul)
                                {
                                    arm_compute::Status status = ClMultiplicationWorkloadValidate(
                                            baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                            baseLayer->GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo(),
                                            activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                            &activationDesc);

                                    if (status)
                                    {
                                        FuseElementwiseBinaryLayer<ElementwiseBinaryLayer>(optimizationViews,
                                                                                           baseLayer,
                                                                                           activationLayer,
                                                                                           activationDesc,
                                                                                           BinaryOperation::Mul,
                                                                                           name);
                                        untouched.erase(baseLayer->GetGuid());
                                        untouched.erase(activationLayer->GetGuid());
                                    }
                                }
                                else if (baseLayer->GetParameters().m_Operation == BinaryOperation::Sub)
                                {
                                    arm_compute::Status status = ClSubtractionValidate(
                                            baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                            baseLayer->GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo(),
                                            activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                            &activationDesc);

                                    if (status)
                                    {
                                        FuseElementwiseBinaryLayer<ElementwiseBinaryLayer>(optimizationViews,
                                                                                           baseLayer,
                                                                                           activationLayer,
                                                                                           activationDesc,
                                                                                           BinaryOperation::Sub,
                                                                                           name);
                                    }
                                }
                                // No fusion available for other BinaryOperations
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

} // namespace armnn
