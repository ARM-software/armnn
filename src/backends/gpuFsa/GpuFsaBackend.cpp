//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GpuFsaBackend.hpp"
#include "GpuFsaBackendContext.hpp"
#include "GpuFsaBackendDefaultAllocator.hpp"
#include "GpuFsaBackendId.hpp"
#include "GpuFsaLayerSupport.hpp"
#include "GpuFsaTensorHandleFactory.hpp"
#include "GpuFsaWorkloadFactory.hpp"

#include <armnn/backends/IBackendContext.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <aclCommon/BaseMemoryManager.hpp>
#include <backendsCommon/SubgraphUtils.hpp>
#include <Optimizer.hpp>

#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/runtime/CL/CLBufferAllocator.h>

#include "layers/GpuFsaBatchMatMul.hpp"
#include "layers/GpuFsaCast.hpp"
#include "layers/GpuFsaConvolution2d.hpp"
#include "layers/GpuFsaDepthwiseConvolution2d.hpp"
#include "layers/GpuFsaElementwiseBinary.hpp"
#include "layers/GpuFsaPooling2d.hpp"
#include "layers/GpuFsaResize.hpp"

namespace armnn
{

template <typename T>
inline void DeleteAsType(const void* const blob)
{
    delete static_cast<const T*>(blob);
}

inline SubgraphView::InputSlots CreateInputsFrom(Layer* layer)
{
    SubgraphView::InputSlots result;
    for (auto&& it = layer->BeginInputSlots(); it != layer->EndInputSlots(); ++it)
    {
        result.push_back(&(*it));
    }
    return result;
}

inline SubgraphView::OutputSlots CreateOutputsFrom(Layer* layer)
{
    SubgraphView::OutputSlots result;
    for (auto&& it = layer->BeginOutputSlots(); it != layer->EndOutputSlots(); ++it)
    {
        result.push_back(&(*it));
    }
    return result;
}

inline SubgraphView::SubgraphViewPtr CreateSubgraphViewFrom(SubgraphView::InputSlots&& inputs,
                                                            SubgraphView::OutputSlots&& outputs,
                                                            SubgraphView::Layers&& layers)
{
    return std::make_unique<SubgraphView>(std::move(inputs), std::move(outputs), std::move(layers));
}

const BackendId& GpuFsaBackend::GetIdStatic()
{
    static const BackendId s_Id{GpuFsaBackendId()};
    return s_Id;
}

IBackendInternal::IMemoryManagerUniquePtr GpuFsaBackend::CreateMemoryManager() const
{
    if (m_UsingCustomAllocator)
    {
        return std::make_unique<GpuFsaMemoryManager>(m_CustomAllocator);
    }
    return std::make_unique<GpuFsaMemoryManager>(std::make_unique<arm_compute::CLBufferAllocator>());
}

IBackendInternal::IWorkloadFactoryPtr GpuFsaBackend::CreateWorkloadFactory(
    const IBackendInternal::IMemoryManagerSharedPtr& memoryManager) const
{
    return std::make_unique<GpuFsaWorkloadFactory>(PolymorphicPointerDowncast<GpuFsaMemoryManager>(memoryManager));
}

IBackendInternal::IWorkloadFactoryPtr GpuFsaBackend::CreateWorkloadFactory(
    TensorHandleFactoryRegistry& registry) const
{
    std::shared_ptr<GpuFsaMemoryManager> memoryManager;
    if (m_UsingCustomAllocator)
    {
        memoryManager = std::make_shared<GpuFsaMemoryManager>(m_CustomAllocator);
    }
    else
    {
        memoryManager = std::make_shared<GpuFsaMemoryManager>(std::make_unique<arm_compute::CLBufferAllocator>());
    }

    std::unique_ptr<ITensorHandleFactory> factory = std::make_unique<GpuFsaTensorHandleFactory>(memoryManager);

    registry.RegisterMemoryManager(memoryManager);
    registry.RegisterFactory(std::move(factory));

    return std::make_unique<GpuFsaWorkloadFactory>(PolymorphicPointerDowncast<GpuFsaMemoryManager>(memoryManager));
}

IBackendInternal::IWorkloadFactoryPtr GpuFsaBackend::CreateWorkloadFactory(
    TensorHandleFactoryRegistry& registry,
    const ModelOptions&,
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

    std::shared_ptr<GpuFsaMemoryManager> memoryManager;
    if (m_UsingCustomAllocator)
    {
        memoryManager = std::make_shared<GpuFsaMemoryManager>(m_CustomAllocator);
    }
    else
    {
        memoryManager = std::make_shared<GpuFsaMemoryManager>(std::make_unique<arm_compute::CLBufferAllocator>());
    }

    std::unique_ptr<ITensorHandleFactory> factory = std::make_unique<GpuFsaTensorHandleFactory>(memoryManager);

    registry.RegisterMemoryManager(memoryManager);
    registry.RegisterFactory(std::move(factory));

    return std::make_unique<GpuFsaWorkloadFactory>(PolymorphicPointerDowncast<GpuFsaMemoryManager>(memoryManager));
}

std::vector<ITensorHandleFactory::FactoryId> GpuFsaBackend::GetHandleFactoryPreferences() const
{
    return std::vector<ITensorHandleFactory::FactoryId> { GpuFsaTensorHandleFactory::GetIdStatic() };
}

void GpuFsaBackend::RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry)
{
    std::shared_ptr<GpuFsaMemoryManager> memoryManager;
    if (m_UsingCustomAllocator)
    {
        memoryManager = std::make_shared<GpuFsaMemoryManager>(m_CustomAllocator);
    }
    else
    {
        memoryManager = std::make_shared<GpuFsaMemoryManager>(std::make_unique<arm_compute::CLBufferAllocator>());
    }

    std::unique_ptr<ITensorHandleFactory> factory = std::make_unique<GpuFsaTensorHandleFactory>(memoryManager);
    registry.RegisterMemoryManager(memoryManager);
    registry.RegisterFactory(std::move(factory));

}

void GpuFsaBackend::RegisterTensorHandleFactories(TensorHandleFactoryRegistry& registry,
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

    std::shared_ptr<GpuFsaMemoryManager> memoryManager;
    if (m_UsingCustomAllocator)
    {
        memoryManager = std::make_shared<GpuFsaMemoryManager>(m_CustomAllocator);
    }
    else
    {
        memoryManager = std::make_shared<GpuFsaMemoryManager>(std::make_unique<arm_compute::CLBufferAllocator>());
    }

    std::unique_ptr<ITensorHandleFactory> factory = std::make_unique<GpuFsaTensorHandleFactory>(memoryManager);
    registry.RegisterMemoryManager(memoryManager);
    registry.RegisterFactory(std::move(factory));
}

IBackendInternal::IBackendContextPtr GpuFsaBackend::CreateBackendContext(const IRuntime::CreationOptions& options) const
{
    return IBackendContextPtr{new GpuFsaBackendContext{options}};
}

IBackendInternal::IBackendProfilingContextPtr GpuFsaBackend::CreateBackendProfilingContext(
    const IRuntime::CreationOptions&, IBackendProfilingPtr&)
{
    return IBackendProfilingContextPtr{};
}

IBackendInternal::ILayerSupportSharedPtr GpuFsaBackend::GetLayerSupport() const
{
    static ILayerSupportSharedPtr layerSupport{new GpuFsaLayerSupport};
    return layerSupport;
}

std::unique_ptr<ICustomAllocator> GpuFsaBackend::GetDefaultAllocator() const
{
    return std::make_unique<GpuFsaBackendDefaultAllocator>();
}

OptimizationViews GpuFsaBackend::OptimizeSubgraphView(const SubgraphView& subgraph,
                                                      const ModelOptions& modelOptions) const
{
    OptimizationViews optimizationViews(modelOptions);

    using namespace arm_compute::experimental::dynamic_fusion;

    auto it = subgraph.end();
    std::map<LayerGuid, Layer*> untouched;
    while (it != subgraph.begin())
    {
        --it;
        Layer& base = *(PolymorphicDowncast<Layer*>(*it));
        untouched.insert({base.GetGuid(), &base});
    }

    GpuFsaLayerSupport supportChecker;
    it = subgraph.end();
    arm_compute::CLCompileContext* compileCtx = &(arm_compute::CLKernelLibrary::get().get_compile_context());

    // Setup the GpuWokloadContext which will exist for the lifetime of the Graph. This contains the TensorInfos
    std::shared_ptr<GpuWorkloadContext> workloadContext = std::make_shared<GpuWorkloadContext>(compileCtx);
    while (it != subgraph.begin())
    {
        --it;
        Layer& base = *(PolymorphicDowncast<Layer*>(*it));
        // Create a GpuFsaPreCompiledBlob, this contains all of the information needed to execute an operator
        GpuFsaPreCompiledBlob* preCompiledBlobPtr = new GpuFsaPreCompiledBlob();
        preCompiledBlobPtr->workloadContext = workloadContext;
        preCompiledBlobPtr->sketch = std::make_unique<GpuWorkloadSketch>(workloadContext.get());

        // Configure and setup the sketch for each supported op. Their data will be wrapped into a PreCompiled layer
        switch (base.GetType())
        {
            case (LayerType::Cast):
            {
                auto input  = base.GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
                auto output = base.GetOutputSlot(0).GetTensorInfo();
                GpuFsaCastCreateOp(preCompiledBlobPtr, input, output);
                break;
            }
            case (LayerType::Convolution2d):
            {
                auto input = base.GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
                auto weights = base.GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo();

                auto desc = PolymorphicDowncast<const Convolution2dDescriptor*>(&base.GetParameters());
                if (desc->m_BiasEnabled)
                {
                    auto bias = base.GetInputSlot(2).GetConnectedOutputSlot()->GetTensorInfo();
                    GpuFsaConvolution2dCreateOp(preCompiledBlobPtr,
                                                input,
                                                *desc,
                                                weights,
                                                bias);
                }
                else
                {
                    GpuFsaConvolution2dCreateOp(preCompiledBlobPtr,
                                                input,
                                                *desc,
                                                weights,
                                                EmptyOptional());
                }
                break;
            }
            case (LayerType::BatchMatMul):
            {
                auto input0 = base.GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
                auto input1 = base.GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo();
                auto desc = PolymorphicDowncast<const BatchMatMulDescriptor*>(&base.GetParameters());
                GpuFsaBatchMatMulCreateOp(preCompiledBlobPtr, input0, input1, *desc);
                break;
            }
            case (LayerType::DepthwiseConvolution2d):
            {
                auto input = base.GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
                auto weights = base.GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo();

                auto desc = PolymorphicDowncast<const DepthwiseConvolution2dDescriptor*>(&base.GetParameters());
                if (desc->m_BiasEnabled)
                {
                    auto bias = base.GetInputSlot(2).GetConnectedOutputSlot()->GetTensorInfo();
                    GpuFsaDepthwiseConvolution2dCreateOp(preCompiledBlobPtr,
                                                         input,
                                                         *desc,
                                                         weights,
                                                         bias);
                }
                else
                {
                    GpuFsaDepthwiseConvolution2dCreateOp(preCompiledBlobPtr,
                                                         input,
                                                         *desc,
                                                         weights,
                                                         EmptyOptional());
                }
                break;
            }
            case LayerType::ElementwiseBinary:
            {
                auto desc = PolymorphicDowncast<const ElementwiseBinaryDescriptor *>(&base.GetParameters());
                auto input0 = base.GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
                auto input1 = base.GetInputSlot(1).GetConnectedOutputSlot()->GetTensorInfo();
                GpuFsaElementwiseBinaryCreateOp(preCompiledBlobPtr, input0, input1, *desc);
                break;
            }
            case (LayerType::Pooling2d):
            {
                auto input = base.GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
                auto desc = PolymorphicDowncast<const Pooling2dDescriptor*>(&base.GetParameters());
                GpuFsaPooling2dCreateOp(preCompiledBlobPtr, input, *desc);
                break;
            }
            case (LayerType::Resize):
            {
                auto input = base.GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo();
                auto desc = PolymorphicDowncast<const ResizeDescriptor*>(&base.GetParameters());
                GpuFsaResizeCreateOp(preCompiledBlobPtr, input, *desc);
                break;
            }
            default:
                // unsupported layer for GpuFsa backend
                continue;
        }

        auto compiledBlob =
            std::make_unique<PreCompiledObjectPtr>(preCompiledBlobPtr, DeleteAsType<GpuFsaPreCompiledBlob>);

        IConnectableLayer* preCompiledLayer = optimizationViews.GetINetwork()->AddPrecompiledLayer(
            PreCompiledDescriptor(base.GetNumInputSlots(), base.GetNumOutputSlots()),
            std::move(*compiledBlob),
            armnn::Optional<BackendId>(GetId()),
            "GpuFsa_Pre_Compiled_Layer");

        // Copy the output tensor infos from sub-graph
        for (unsigned int i = 0; i < subgraph.GetNumOutputSlots(); i++)
        {
            preCompiledLayer->GetOutputSlot(i).SetTensorInfo(base.GetOutputSlot(i).GetTensorInfo());
        }

        SubgraphView::SubgraphViewPtr substituteSubgraph =
            CreateSubgraphViewFrom(CreateInputsFrom(&base),
                                   CreateOutputsFrom(&base),
                                   {&base});

        optimizationViews.AddSubstitution({ std::move(*substituteSubgraph), SubgraphView(preCompiledLayer) });

        untouched.erase(base.GetGuid());
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
