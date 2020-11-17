//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBackend.hpp"
#include "ClBackendId.hpp"
#include "ClBackendModelContext.hpp"
#include "ClWorkloadFactory.hpp"
#include "ClBackendContext.hpp"
#include "ClLayerSupport.hpp"
#include "ClTensorHandleFactory.hpp"

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
#include "workloads/ClDivisionFloatWorkload.hpp"
#include "workloads/ClFullyConnectedWorkload.hpp"
#include "workloads/ClMultiplicationWorkload.hpp"
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
    auto memoryManager = std::make_shared<ClMemoryManager>(std::make_unique<arm_compute::CLBufferAllocator>());

    registry.RegisterMemoryManager(memoryManager);
    registry.RegisterFactory(std::make_unique<ClTensorHandleFactory>(memoryManager));

    return std::make_unique<ClWorkloadFactory>(
            PolymorphicPointerDowncast<ClMemoryManager>(memoryManager));
}

IBackendInternal::IWorkloadFactoryPtr ClBackend::CreateWorkloadFactory(
    TensorHandleFactoryRegistry& registry, const ModelOptions& modelOptions) const
{
    auto memoryManager = std::make_shared<ClMemoryManager>(std::make_unique<arm_compute::CLBufferAllocator>());

    registry.RegisterMemoryManager(memoryManager);
    registry.RegisterFactory(std::make_unique<ClTensorHandleFactory>(memoryManager));

    return std::make_unique<ClWorkloadFactory>(
        PolymorphicPointerDowncast<ClMemoryManager>(memoryManager), CreateBackendSpecificModelContext(modelOptions));
}

std::vector<ITensorHandleFactory::FactoryId> ClBackend::GetHandleFactoryPreferences() const
{
    return std::vector<ITensorHandleFactory::FactoryId> {ClTensorHandleFactory::GetIdStatic()};
}

void ClBackend::RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry)
{
    auto mgr = std::make_shared<ClMemoryManager>(std::make_unique<arm_compute::CLBufferAllocator>());

    registry.RegisterMemoryManager(mgr);
    registry.RegisterFactory(std::make_unique<ClTensorHandleFactory>(mgr));
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

IBackendInternal::Optimizations ClBackend::GetOptimizations() const
{
    return Optimizations{};
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

OptimizationViews ClBackend::OptimizeSubgraphView(const SubgraphView& subgraph,
                                                  const ModelOptions& modelOptions) const
{
    OptimizationViews optimizationViews;

    auto it = subgraph.end();
    bool isFastMathEnabled = false;
    std::map<LayerGuid, Layer*> untouched;

    while (it != subgraph.begin())
    {
        --it;
        Layer& base = **it;
        untouched.insert({base.GetGuid(), &base});
    }

    it = subgraph.end();
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
    while (it != subgraph.begin())
    {
        --it;
        Layer& base = **it;

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
                        if (childInput->GetOwningLayer().GetType() == LayerType::Activation)
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

                                arm_compute::Status status = ClConvolution2dWorkloadValidate(
                                        baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        baseLayer->GetParameters(),
                                        baseLayer->m_Weight->GetTensorInfo(),
                                        biases,
                                        isFastMathEnabled,
                                        &activationDesc);

                                if (status)
                                {
                                    FuseLayerWithWeightsAndBiases<Convolution2dLayer>(optimizationViews,
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

                                arm_compute::Status status = ClDepthwiseConvolutionWorkloadValidate(
                                        baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        baseLayer->GetParameters(),
                                        baseLayer->m_Weight->GetTensorInfo(),
                                        biases,
                                        &activationDesc);

                                if (status)
                                {
                                    FuseLayerWithWeightsAndBiases<DepthwiseConvolution2dLayer>(optimizationViews,
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

                                arm_compute::Status status = ClFullyConnectedWorkloadValidate(
                                        baseLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        activationLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo(),
                                        baseLayer->m_Weight->GetTensorInfo(),
                                        baseLayer->m_Bias->GetTensorInfo(),
                                        baseLayer->GetParameters(),
                                        &activationDesc);

                                if (status)
                                {
                                    FuseLayerWithWeightsAndBiases<FullyConnectedLayer>(optimizationViews,
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
                                            FuseLayerWithParameters<BatchNormalizationLayer>(optimizationViews,
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
                                    FuseLayerWithoutParameters<AdditionLayer>(optimizationViews,
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
                                    FuseLayerWithoutParameters<DivisionLayer>(optimizationViews,
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
                                    FuseLayerWithoutParameters<MultiplicationLayer>(optimizationViews,
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
                                    FuseLayerWithoutParameters<SubtractionLayer>(optimizationViews,
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
