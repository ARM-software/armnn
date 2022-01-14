//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn_delegate.hpp>

#include "Version.hpp"

#include "Activation.hpp"
#include "ArgMinMax.hpp"
#include "BatchSpace.hpp"
#include "Comparison.hpp"
#include "Convolution.hpp"
#include "Control.hpp"
#include "ElementwiseBinary.hpp"
#include "ElementwiseUnary.hpp"
#include "Fill.hpp"
#include "FullyConnected.hpp"
#include "Gather.hpp"
#include "LogicalBinary.hpp"
#include "Lstm.hpp"
#include "Normalization.hpp"
#include "Pack.hpp"
#include "Pad.hpp"
#include "Pooling.hpp"
#include "Prelu.hpp"
#include "Quantization.hpp"
#include "Redefine.hpp"
#include "Reduce.hpp"
#include "Resize.hpp"
#include "Round.hpp"
#include "Shape.hpp"
#include "Slice.hpp"
#include "Softmax.hpp"
#include "SpaceDepth.hpp"
#include "Split.hpp"
#include "Transpose.hpp"
#include "UnidirectionalSequenceLstm.hpp"
#include "Unpack.hpp"

#include <armnnUtils/Filesystem.hpp>
#include <armnn/utility/Timer.hpp>
#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/context_util.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <algorithm>
#include <iostream>
#include <sstream>

namespace armnnDelegate
{

DelegateOptions TfLiteArmnnDelegateOptionsDefault()
{
    DelegateOptions options(armnn::Compute::CpuRef);
    return options;
}

TfLiteDelegate* TfLiteArmnnDelegateCreate(armnnDelegate::DelegateOptions options)
{
    auto* armnnDelegate = new ::armnnDelegate::Delegate(options);
    return armnnDelegate->GetDelegate();
}

void TfLiteArmnnDelegateDelete(TfLiteDelegate* tfLiteDelegate)
{
    if (tfLiteDelegate != nullptr)
    {
        delete static_cast<::armnnDelegate::Delegate*>(tfLiteDelegate->data_);
    }
}

TfLiteStatus DoPrepare(TfLiteContext* tfLiteContext, TfLiteDelegate* tfLiteDelegate)
{
    TfLiteIntArray* supportedOperators =
        static_cast<::armnnDelegate::Delegate*>(tfLiteDelegate->data_)->IdentifyOperatorsToDelegate(tfLiteContext);

    // ArmNN Delegate Registration
    static const TfLiteRegistration kArmnnSubgraphRegistration = {
        // ArmnnSubgraph Init
        .init = [](TfLiteContext* tfLiteContext, const char* buffer, size_t length) -> void* {
            armnn::IgnoreUnused(length);
            const TfLiteDelegateParams* parameters = reinterpret_cast<const TfLiteDelegateParams*>(buffer);

            return static_cast<void*>(ArmnnSubgraph::Create(
                tfLiteContext, parameters, static_cast<::armnnDelegate::Delegate*>(parameters->delegate->data_)));
        },
        // ArmnnSubgraph Free
        .free = [](TfLiteContext* tfLiteContext, void* buffer) -> void {
            armnn::IgnoreUnused(tfLiteContext);
            if (buffer != nullptr)
            {
                delete static_cast<ArmnnSubgraph*>(buffer);
            }
        },
        // ArmnnSubgraph Prepare
        .prepare = [](TfLiteContext* tfLiteContext, TfLiteNode* tfLiteNode) -> TfLiteStatus {
            if (tfLiteNode->user_data == nullptr)
            {
                return kTfLiteError;
            }
            return static_cast<ArmnnSubgraph*>(tfLiteNode->user_data)->Prepare(tfLiteContext);
        },
        // ArmnnSubgraph Invoke
        .invoke = [](TfLiteContext* tfLiteContext, TfLiteNode* tfLiteNode) -> TfLiteStatus {
            if (tfLiteNode->user_data == nullptr)
            {
                return kTfLiteError;
            }

            return static_cast<ArmnnSubgraph*>(tfLiteNode->user_data)->Invoke(tfLiteContext, tfLiteNode);
        },

        .profiling_string = nullptr,
        .builtin_code = kTfLiteBuiltinDelegate,
        .custom_name = "TfLiteArmNnDelegate",
        .version = 1,
    };

    const TfLiteStatus status =
        tfLiteContext->ReplaceNodeSubsetsWithDelegateKernels(
            tfLiteContext, kArmnnSubgraphRegistration, supportedOperators, tfLiteDelegate);

    TfLiteIntArrayFree(supportedOperators);
    return status;

}

Delegate::Delegate(armnnDelegate::DelegateOptions options)
  : m_Runtime(nullptr, nullptr),
    m_Options(std::move(options))
{
    // Configures logging for ARMNN
    if (options.IsLoggingEnabled())
    {
        armnn::ConfigureLogging(true, true, options.GetLoggingSeverity());
    }

    // Create ArmNN Runtime
    m_Runtime = armnn::IRuntime::Create(options.GetRuntimeOptions());

    std::vector<armnn::BackendId> backends;
    if (m_Runtime)
    {
        const armnn::BackendIdSet supportedDevices = m_Runtime->GetDeviceSpec().GetSupportedBackends();
        for (auto& backend : m_Options.GetBackends())
        {
            if (std::find(supportedDevices.cbegin(), supportedDevices.cend(), backend) == supportedDevices.cend())
            {
                TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                    "TfLiteArmnnDelegate: Requested unknown backend %s", backend.Get().c_str());
            }
            else
            {
                backends.push_back(backend);
            }
        }
    }

    if (backends.empty())
    {
        // No known backend specified
        throw armnn::InvalidArgumentException("TfLiteArmnnDelegate: No known backend specified.");
    }
    m_Options.SetBackends(backends);

    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO, "TfLiteArmnnDelegate: Created TfLite ArmNN delegate.");
}

TfLiteIntArray* Delegate::IdentifyOperatorsToDelegate(TfLiteContext* tfLiteContext)
{
    TfLiteIntArray* executionPlan = nullptr;
    if (tfLiteContext->GetExecutionPlan(tfLiteContext, &executionPlan) != kTfLiteOk)
    {
        TF_LITE_KERNEL_LOG(tfLiteContext, "TfLiteArmnnDelegate: Unable to get graph execution plan.");
        return nullptr;
    }

    // Delegate data with null network
    DelegateData delegateData(m_Options.GetBackends());

    TfLiteIntArray* nodesToDelegate = TfLiteIntArrayCreate(executionPlan->size);
    nodesToDelegate->size = 0;
    for (int i = 0; i < executionPlan->size; ++i)
    {
        const int nodeIndex = executionPlan->data[i];

        // If TfLite nodes can be delegated to ArmNN
        TfLiteNode* tfLiteNode = nullptr;
        TfLiteRegistration* tfLiteRegistration = nullptr;
        if (tfLiteContext->GetNodeAndRegistration(
            tfLiteContext, nodeIndex, &tfLiteNode, &tfLiteRegistration) != kTfLiteOk)
        {
            TF_LITE_KERNEL_LOG(tfLiteContext,
                               "TfLiteArmnnDelegate: Unable to get node and registration for node %d.",
                               nodeIndex);
            continue;
        }

        if (ArmnnSubgraph::VisitNode(
                   delegateData, tfLiteContext, tfLiteRegistration, tfLiteNode, nodeIndex) != kTfLiteOk)
        {
            // node is not supported by ArmNN
            continue;
        }

        nodesToDelegate->data[nodesToDelegate->size++] = nodeIndex;
    }

    std::sort(&nodesToDelegate->data[0], &nodesToDelegate->data[nodesToDelegate->size]);
    return nodesToDelegate;
}

TfLiteDelegate* Delegate::GetDelegate()
{
    return &m_Delegate;
}

const std::string Delegate::GetVersion()
{
    return DELEGATE_VERSION;
}

TfLiteStatus ArmnnSubgraph::AddInputLayer(DelegateData& delegateData,
                                          TfLiteContext* tfLiteContext,
                                          const TfLiteIntArray* inputs,
                                          std::vector<armnn::BindingPointInfo>& inputBindings)
{
    const size_t numInputs = static_cast<size_t>(inputs->size);
    for (unsigned int i = 0; i < numInputs; ++i)
    {
        const int32_t tensorId = inputs->data[i];
        const TfLiteTensor tensor = tfLiteContext->tensors[tensorId];
        // Do not create bindings for constant inputs
        if (tensor.allocation_type == kTfLiteMmapRo)
        {
            continue;
        }

        auto bindingId = static_cast<armnn::LayerBindingId>((tensorId));
        armnn::IConnectableLayer* layer = delegateData.m_Network->AddInputLayer(bindingId);

        auto tensorInfo = GetTensorInfoForTfLiteTensor(tensor);
        armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
        outputSlot.SetTensorInfo(tensorInfo);

        // Store for creating connections
        delegateData.m_OutputSlotForNode[static_cast<unsigned long>(tensorId)] = &outputSlot;

        inputBindings.push_back(std::make_pair(bindingId, tensorInfo));
    }

    return kTfLiteOk;
}

TfLiteStatus ArmnnSubgraph::AddOutputLayer(DelegateData& delegateData,
                                           TfLiteContext* tfLiteContext,
                                           const TfLiteIntArray* outputs,
                                           std::vector<armnn::BindingPointInfo>& outputBindings)
{
    const size_t numOutputs = static_cast<size_t>(outputs->size);
    for (unsigned int i = 0; i < numOutputs; ++i)
    {
        const int32_t tensorId = outputs->data[i];
        const TfLiteTensor tensor = tfLiteContext->tensors[tensorId];

        auto bindingId = static_cast<armnn::LayerBindingId>((tensorId));
        armnn::IConnectableLayer* layer = delegateData.m_Network->AddOutputLayer(bindingId);

        auto tensorInfo = GetTensorInfoForTfLiteTensor(tensor);
        ARMNN_ASSERT(delegateData.m_OutputSlotForNode[static_cast<unsigned long>(tensorId)] != nullptr);
        delegateData.m_OutputSlotForNode[static_cast<unsigned long>(tensorId)]->Connect(layer->GetInputSlot(0));
        outputBindings.push_back(std::make_pair(bindingId, tensorInfo));
    }

    return kTfLiteOk;
}

ArmnnSubgraph* ArmnnSubgraph::Create(TfLiteContext* tfLiteContext,
                                     const TfLiteDelegateParams* parameters,
                                     const Delegate* delegate)
{
    const auto startTime = armnn::GetTimeNow();
    ARMNN_LOG(info) << "ArmnnSubgraph creation";

    TfLiteIntArray* executionPlan;
    if (tfLiteContext->GetExecutionPlan(tfLiteContext, &executionPlan) != kTfLiteOk)
    {
        return nullptr;
    }

    // Initialize DelegateData holds network and output slots information
    DelegateData delegateData(delegate->m_Options.GetBackends());

    // Build ArmNN Network
    armnn::NetworkOptions networkOptions = {};
    armnn::NetworkId networkId;
    delegateData.m_Network = armnn::INetwork::Create(networkOptions);

    delegateData.m_OutputSlotForNode = std::vector<armnn::IOutputSlot*>(tfLiteContext->tensors_size, nullptr);

    std::vector<armnn::BindingPointInfo> inputBindings;
    std::vector<armnn::BindingPointInfo> outputBindings;

    // Add input layer
    auto status = AddInputLayer(delegateData, tfLiteContext, parameters->input_tensors, inputBindings);
    if (status != kTfLiteOk)
    {
        throw armnn::Exception("TfLiteArmnnDelegate: Unable to add Inputs to the network!");
    }

    // Parse TfLite delegate nodes to ArmNN
    const auto parseStartTime = armnn::GetTimeNow();
    for (int i = 0; i < parameters->nodes_to_replace->size; ++i)
    {
        const int nodeIndex = parameters->nodes_to_replace->data[i];

        TfLiteNode* tfLiteNode = nullptr;
        TfLiteRegistration* tfLiteRegistration = nullptr;
        if (tfLiteContext->GetNodeAndRegistration(
            tfLiteContext, nodeIndex, &tfLiteNode, &tfLiteRegistration) != kTfLiteOk)
        {
            throw armnn::Exception(&"TfLiteArmnnDelegate: Unable to get node registration: " [ nodeIndex]);
        }

        if (VisitNode(delegateData, tfLiteContext, tfLiteRegistration, tfLiteNode, nodeIndex) != kTfLiteOk)
        {
            throw armnn::Exception(&"TfLiteArmnnDelegate: Unable to parse node: " [ nodeIndex]);
        }
    }
    ARMNN_LOG(info) << "Parse nodes to ArmNN time: " << std::setprecision(2)
                    << std::fixed << armnn::GetTimeDuration(parseStartTime).count() << " ms";

    // Add Output layer
    status = AddOutputLayer(delegateData, tfLiteContext, parameters->output_tensors, outputBindings);
    if (status != kTfLiteOk)
    {
        throw armnn::Exception("TfLiteArmnnDelegate: Unable to add Outputs to the network!");
    }

    // Optimize ArmNN network
    armnn::IOptimizedNetworkPtr optNet(nullptr, nullptr);
    try
    {
        const auto optimizeStartTime = armnn::GetTimeNow();
        optNet = armnn::Optimize(*(delegateData.m_Network.get()),
                                 delegate->m_Options.GetBackends(),
                                 delegate->m_Runtime->GetDeviceSpec(),
                                 delegate->m_Options.GetOptimizerOptions());
        ARMNN_LOG(info) << "Optimize ArmnnSubgraph time: " << std::setprecision(2)
                        << std::fixed << armnn::GetTimeDuration(optimizeStartTime).count() << " ms";
    }
    catch (std::exception &ex)
    {
        std::stringstream exMessage;
        exMessage << "TfLiteArmnnDelegate: Exception (" << ex.what() << ") caught from optimize.";
        throw armnn::Exception(exMessage.str());
    }
    if (!optNet)
    {
        // Optimize failed
        throw armnn::Exception("TfLiteArmnnDelegate: Unable to optimize the network!");
    }

    // If set, we will serialize the optimized model into a dot file.
    const std::string serializeToDotFile = delegate->m_Options.GetSerializeToDot();
    if (!serializeToDotFile.empty())
    {
        ARMNN_LOG(info) << "Writing graph to dot file: " << serializeToDotFile;
        fs::path filename = serializeToDotFile;
        std::fstream file(filename.c_str(), std::ios_base::out);
        optNet->SerializeToDot(file);
    }

    try
    {
        const auto loadStartTime = armnn::GetTimeNow();

        // Load graph into runtime
        std::string errorMessage;
        armnn::Status loadingStatus;
        armnn::MemorySource memorySource = armnn::MemorySource::Undefined;
        if (delegate->m_Options.GetOptimizerOptions().m_ImportEnabled)
        {
            memorySource = armnn::MemorySource::Malloc;
        }
        armnn::INetworkProperties networkProperties(false,
                                                    memorySource,
                                                    memorySource,
                                                    delegate->m_Options.GetInternalProfilingState(),
                                                    delegate->m_Options.GetInternalProfilingDetail());
        loadingStatus = delegate->m_Runtime->LoadNetwork(networkId,
                                                         std::move(optNet),
                                                         errorMessage,
                                                         networkProperties);
        if (loadingStatus != armnn::Status::Success)
        {
            // Network load failed.
            throw armnn::Exception("TfLiteArmnnDelegate: Network could not be loaded:" + errorMessage);
        }

        ARMNN_LOG(info) << "Load ArmnnSubgraph time: " << std::setprecision(2)
                        << std::fixed << armnn::GetTimeDuration(loadStartTime).count() << " ms";
    }
    catch (std::exception& ex)
    {
        std::stringstream exMessage;
        exMessage << "TfLiteArmnnDelegate: Exception (" << ex.what() << ") caught from LoadNetwork.";
        throw armnn::Exception(exMessage.str());
    }

    // Register debug callback function
    if (delegate->m_Options.GetDebugCallbackFunction().has_value())
    {
        delegate->m_Runtime->RegisterDebugCallback(networkId, delegate->m_Options.GetDebugCallbackFunction().value());
    }

    ARMNN_LOG(info) << "Overall ArmnnSubgraph creation time: " << std::setprecision(2)
                    << std::fixed << armnn::GetTimeDuration(startTime).count() << " ms\n";

    // Create a new SubGraph with networkId and runtime
    return new ArmnnSubgraph(networkId, delegate->m_Runtime.get(), inputBindings, outputBindings);
}

TfLiteStatus ArmnnSubgraph::Prepare(TfLiteContext* tfLiteContext)
{
    armnn::IgnoreUnused(tfLiteContext);
    return kTfLiteOk;
}

TfLiteStatus ArmnnSubgraph::Invoke(TfLiteContext* tfLiteContext, TfLiteNode* tfLiteNode)
{
    // Prepare inputs
    armnn::InputTensors inputTensors;
    size_t inputIndex = 0;
    for (auto inputIdx : tflite::TfLiteIntArrayView(tfLiteNode->inputs))
    {
        TfLiteTensor* tensor = &tfLiteContext->tensors[inputIdx];
        if (tensor->allocation_type != kTfLiteMmapRo)
        {
            const armnn::BindingPointInfo& inputBinding = m_InputBindings[inputIndex];
            armnn::TensorInfo inputTensorInfo = inputBinding.second;
            inputTensorInfo.SetConstant(true);
            const armnn::ConstTensor inputTensor(inputTensorInfo, tensor->data.data);
            inputTensors.emplace_back(inputIdx, inputTensor);

            ++inputIndex;
        }
    }

    // Prepare outputs
    armnn::OutputTensors outputTensors;
    size_t outputIndex = 0;
    for (auto outputIdx : tflite::TfLiteIntArrayView(tfLiteNode->outputs))
    {
        const armnn::BindingPointInfo& outputBinding = m_OutputBindings[outputIndex];
        TfLiteTensor* tensor = &tfLiteContext->tensors[outputIdx];
        const armnn::Tensor outputTensor(outputBinding.second, tensor->data.data);
        outputTensors.emplace_back(outputIdx, outputTensor);

        ++outputIndex;
    }

    // Run graph
    auto status = m_Runtime->EnqueueWorkload(m_NetworkId, inputTensors, outputTensors);
    // The delegate holds its own Arm NN runtime so this is our last chance to print internal profiling data.
    std::shared_ptr<armnn::IProfiler> profiler = m_Runtime->GetProfiler(m_NetworkId);
    if (profiler && profiler->IsProfilingEnabled())
    {
        profiler->Print(std::cout);
    }
    return (status == armnn::Status::Success) ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ArmnnSubgraph::VisitNode(DelegateData& delegateData,
                                      TfLiteContext* tfLiteContext,
                                      TfLiteRegistration* tfLiteRegistration,
                                      TfLiteNode* tfLiteNode,
                                      int nodeIndex)
{
    switch (tfLiteRegistration->builtin_code)
    {
        case kTfLiteBuiltinAbs:
            return VisitElementwiseUnaryOperator(delegateData,
                                                 tfLiteContext,
                                                 tfLiteNode,
                                                 nodeIndex,
                                                 armnn::UnaryOperation::Abs);
        case kTfLiteBuiltinAdd:
            return VisitElementwiseBinaryOperator(delegateData,
                                                  tfLiteContext,
                                                  tfLiteNode,
                                                  nodeIndex,
                                                  kTfLiteBuiltinAdd);
        case kTfLiteBuiltinArgMax:
            return VisitArgMinMaxOperator(delegateData,
                                          tfLiteContext,
                                          tfLiteNode,
                                          nodeIndex,
                                          kTfLiteBuiltinArgMax);
        case kTfLiteBuiltinArgMin:
            return VisitArgMinMaxOperator(delegateData,
                                          tfLiteContext,
                                          tfLiteNode,
                                          nodeIndex,
                                          kTfLiteBuiltinArgMin);
        case kTfLiteBuiltinAveragePool2d:
            return VisitPoolingOperator(delegateData,
                                        tfLiteContext,
                                        tfLiteNode,
                                        nodeIndex,
                                        kTfLiteBuiltinAveragePool2d);
        case kTfLiteBuiltinBatchToSpaceNd:
            return VisitBatchToSpaceNdOperator(delegateData,
                                               tfLiteContext,
                                               tfLiteNode,
                                               nodeIndex,
                                               kTfLiteBuiltinBatchToSpaceNd);
        case kTfLiteBuiltinCast:
            return VisitCastOperator(delegateData,
                                     tfLiteContext,
                                     tfLiteNode,
                                     nodeIndex,
                                     kTfLiteBuiltinCast);
        case kTfLiteBuiltinConcatenation:
            return VisitControlOperator(delegateData,
                                        tfLiteContext,
                                        tfLiteNode,
                                        nodeIndex,
                                        kTfLiteBuiltinConcatenation);
        case kTfLiteBuiltinConv2d:
            return VisitConvolutionOperator(delegateData,
                                            tfLiteContext,
                                            tfLiteNode,
                                            nodeIndex,
                                            kTfLiteBuiltinConv2d);
// Conv3d is only correctly supported for external delegates from TF Lite v2.6, as there was a breaking bug in v2.5.
#if defined(ARMNN_POST_TFLITE_2_5)
        case kTfLiteBuiltinConv3d:
            return VisitConvolutionOperator(delegateData,
                                            tfLiteContext,
                                            tfLiteNode,
                                            nodeIndex,
                                            kTfLiteBuiltinConv3d);
#endif
        case kTfLiteBuiltinDepthToSpace:
            return VisitDepthToSpaceOperator(delegateData,
                                             tfLiteContext,
                                             tfLiteNode,
                                             nodeIndex,
                                             kTfLiteBuiltinDepthToSpace);
        case kTfLiteBuiltinDepthwiseConv2d:
            return VisitConvolutionOperator(delegateData,
                                            tfLiteContext,
                                            tfLiteNode,
                                            nodeIndex,
                                            kTfLiteBuiltinDepthwiseConv2d);
        case kTfLiteBuiltinDequantize:
            return VisitDequantizeOperator(delegateData,
                                           tfLiteContext,
                                           tfLiteNode,
                                           nodeIndex,
                                           kTfLiteBuiltinDequantize);
        case kTfLiteBuiltinDiv:
            return VisitElementwiseBinaryOperator(delegateData,
                                                  tfLiteContext,
                                                  tfLiteNode,
                                                  nodeIndex,
                                                  kTfLiteBuiltinDiv);
        case kTfLiteBuiltinElu:
            return VisitActivationOperator(delegateData,
                                           tfLiteContext,
                                           tfLiteNode,
                                           nodeIndex,
                                           kTfLiteBuiltinElu);
        case kTfLiteBuiltinEqual:
            return VisitComparisonOperator(delegateData,
                                           tfLiteContext,
                                           tfLiteNode,
                                           nodeIndex,
                                           kTfLiteBuiltinEqual);
        case kTfLiteBuiltinExp:
            return VisitElementwiseUnaryOperator(delegateData,
                                                 tfLiteContext,
                                                 tfLiteNode,
                                                 nodeIndex,
                                                 armnn::UnaryOperation::Exp);
        case kTfLiteBuiltinExpandDims:
            return VisitExpandDimsOperator(delegateData,
                                           tfLiteContext,
                                           tfLiteNode,
                                           nodeIndex,
                                           kTfLiteBuiltinExpandDims);
        case kTfLiteBuiltinFill:
            return VisitFillOperator(delegateData,
                                     tfLiteContext,
                                     tfLiteNode,
                                     nodeIndex,
                                     kTfLiteBuiltinFill);
        case kTfLiteBuiltinFloor:
            return VisitFloorOperator(delegateData,
                                      tfLiteContext,
                                      tfLiteNode,
                                      nodeIndex,
                                      kTfLiteBuiltinFloor);
        case kTfLiteBuiltinFloorDiv:
            return VisitElementwiseBinaryOperator(delegateData,
                                                  tfLiteContext,
                                                  tfLiteNode,
                                                  nodeIndex,
                                                  kTfLiteBuiltinFloorDiv);
        case kTfLiteBuiltinFullyConnected:
            return VisitFullyConnectedOperator(delegateData,
                                               tfLiteContext,
                                               tfLiteNode,
                                               nodeIndex,
                                               kTfLiteBuiltinFullyConnected);
        case kTfLiteBuiltinGather:
            return VisitGatherOperator(delegateData,
                                       tfLiteContext,
                                       tfLiteNode,
                                       nodeIndex,
                                       kTfLiteBuiltinGather);
        case kTfLiteBuiltinGreater:
            return VisitComparisonOperator(delegateData,
                                           tfLiteContext,
                                           tfLiteNode,
                                           nodeIndex,
                                           kTfLiteBuiltinGreater);
        case kTfLiteBuiltinGreaterEqual:
            return VisitComparisonOperator(delegateData,
                                           tfLiteContext,
                                           tfLiteNode,
                                           nodeIndex,
                                           kTfLiteBuiltinGreaterEqual);
        case kTfLiteBuiltinHardSwish:
            return VisitActivationOperator(delegateData,
                                           tfLiteContext,
                                           tfLiteNode,
                                           nodeIndex,
                                           kTfLiteBuiltinHardSwish);
        case kTfLiteBuiltinL2Normalization:
            return VisitL2NormalizationOperator(delegateData,
                                                tfLiteContext,
                                                tfLiteNode,
                                                nodeIndex,
                                                kTfLiteBuiltinL2Normalization);
        case kTfLiteBuiltinL2Pool2d:
            return VisitPoolingOperator(delegateData,
                                        tfLiteContext,
                                        tfLiteNode,
                                        nodeIndex,
                                        kTfLiteBuiltinL2Pool2d);
        case kTfLiteBuiltinLess:
            return VisitComparisonOperator(delegateData,
                                           tfLiteContext,
                                           tfLiteNode,
                                           nodeIndex,
                                           kTfLiteBuiltinLess);
        case kTfLiteBuiltinLessEqual:
            return VisitComparisonOperator(delegateData,
                                           tfLiteContext,
                                           tfLiteNode,
                                           nodeIndex,
                                           kTfLiteBuiltinLessEqual);
        case kTfLiteBuiltinLocalResponseNormalization:
            return VisitLocalResponseNormalizationOperator(delegateData,
                                                           tfLiteContext,
                                                           tfLiteNode,
                                                           nodeIndex,
                                                           kTfLiteBuiltinLocalResponseNormalization);
        case kTfLiteBuiltinLogicalAnd:
            return VisitLogicalBinaryOperator(delegateData,
                                              tfLiteContext,
                                              tfLiteNode,
                                              nodeIndex,
                                              kTfLiteBuiltinLogicalAnd,
                                              armnn::LogicalBinaryOperation::LogicalAnd);
        case kTfLiteBuiltinLogicalNot:
            return VisitElementwiseUnaryOperator(delegateData,
                                                 tfLiteContext,
                                                 tfLiteNode,
                                                 nodeIndex,
                                                 armnn::UnaryOperation::LogicalNot);
        case kTfLiteBuiltinLogicalOr:
            return VisitLogicalBinaryOperator(delegateData,
                                              tfLiteContext,
                                              tfLiteNode,
                                              nodeIndex,
                                              kTfLiteBuiltinLogicalOr,
                                              armnn::LogicalBinaryOperation::LogicalOr);
        case kTfLiteBuiltinLogistic:
            return VisitActivationOperator(delegateData,
                                           tfLiteContext,
                                           tfLiteNode,
                                           nodeIndex,
                                           kTfLiteBuiltinLogistic);
        case kTfLiteBuiltinLogSoftmax:
            return VisitSoftmaxOperator(delegateData,
                                        tfLiteContext,
                                        tfLiteNode,
                                        nodeIndex,
                                        kTfLiteBuiltinLogSoftmax);
        case kTfLiteBuiltinLstm:
            return VisitLstmOperator(delegateData,
                                     tfLiteContext,
                                     tfLiteNode,
                                     nodeIndex,
                                     kTfLiteBuiltinLstm);
        case kTfLiteBuiltinMaxPool2d:
            return VisitPoolingOperator(delegateData,
                                        tfLiteContext,
                                        tfLiteNode,
                                        nodeIndex,
                                        kTfLiteBuiltinMaxPool2d);
        case kTfLiteBuiltinMaximum:
            return VisitElementwiseBinaryOperator(delegateData,
                                                  tfLiteContext,
                                                  tfLiteNode,
                                                  nodeIndex,
                                                  kTfLiteBuiltinMaximum);
        case kTfLiteBuiltinMean:
            return VisitControlOperator(delegateData,
                                        tfLiteContext,
                                        tfLiteNode,
                                        nodeIndex,
                                        kTfLiteBuiltinMean);
        case kTfLiteBuiltinMinimum:
            return VisitElementwiseBinaryOperator(delegateData,
                                                  tfLiteContext,
                                                  tfLiteNode,
                                                  nodeIndex,
                                                  kTfLiteBuiltinMinimum);
        case kTfLiteBuiltinMirrorPad:
            return VisitPadOperator(delegateData,
                                    tfLiteContext,
                                    tfLiteNode,
                                    nodeIndex,
                                    kTfLiteBuiltinMirrorPad);
        case kTfLiteBuiltinMul:
            return VisitElementwiseBinaryOperator(delegateData,
                                                  tfLiteContext,
                                                  tfLiteNode,
                                                  nodeIndex,
                                                  kTfLiteBuiltinMul);
        case kTfLiteBuiltinNeg:
            return VisitElementwiseUnaryOperator(delegateData,
                                                 tfLiteContext,
                                                 tfLiteNode,
                                                 nodeIndex,
                                                 armnn::UnaryOperation::Neg);
        case kTfLiteBuiltinNotEqual:
            return VisitComparisonOperator(delegateData,
                                           tfLiteContext,
                                           tfLiteNode,
                                           nodeIndex,
                                           kTfLiteBuiltinNotEqual);
        case kTfLiteBuiltinPack:
            return VisitPackOperator(delegateData,
                                     tfLiteContext,
                                     tfLiteNode,
                                     nodeIndex,
                                     kTfLiteBuiltinPack);
        case kTfLiteBuiltinPad:
            return VisitPadOperator(delegateData,
                                    tfLiteContext,
                                    tfLiteNode,
                                    nodeIndex,
                                    kTfLiteBuiltinPad);
        case kTfLiteBuiltinPadv2:
            return VisitPadOperator(delegateData,
                                    tfLiteContext,
                                    tfLiteNode,
                                    nodeIndex,
                                    kTfLiteBuiltinPadv2);
        case kTfLiteBuiltinPrelu:
            return VisitPreluOperator(delegateData,
                                      tfLiteContext,
                                      tfLiteNode,
                                      nodeIndex,
                                      kTfLiteBuiltinPrelu);
        case kTfLiteBuiltinQuantize:
            return VisitQuantizeOperator(delegateData,
                                         tfLiteContext,
                                         tfLiteNode,
                                         nodeIndex,
                                         kTfLiteBuiltinQuantize);
        case kTfLiteBuiltinRank:
            return VisitControlOperator(delegateData,
                                        tfLiteContext,
                                        tfLiteNode,
                                        nodeIndex,
                                        kTfLiteBuiltinRank);
        case kTfLiteBuiltinReduceMax:
            return VisitReduceOperator(delegateData,
                                       tfLiteContext,
                                       tfLiteNode,
                                       nodeIndex,
                                       kTfLiteBuiltinReduceMax);
        case kTfLiteBuiltinReduceMin:
            return VisitReduceOperator(delegateData,
                                       tfLiteContext,
                                       tfLiteNode,
                                       nodeIndex,
                                       kTfLiteBuiltinReduceMin);
        case kTfLiteBuiltinReduceProd:
            return VisitReduceOperator(delegateData,
                                       tfLiteContext,
                                       tfLiteNode,
                                       nodeIndex,
                                       kTfLiteBuiltinReduceProd);
        case kTfLiteBuiltinRelu:
            return VisitActivationOperator(delegateData,
                                           tfLiteContext,
                                           tfLiteNode,
                                           nodeIndex,
                                           kTfLiteBuiltinRelu);
        case kTfLiteBuiltinReluN1To1:
            return VisitActivationOperator(delegateData,
                                           tfLiteContext,
                                           tfLiteNode,
                                           nodeIndex,
                                           kTfLiteBuiltinReluN1To1);
        case kTfLiteBuiltinRelu6:
            return VisitActivationOperator(delegateData,
                                           tfLiteContext,
                                           tfLiteNode,
                                           nodeIndex,
                                           kTfLiteBuiltinRelu6);
        case kTfLiteBuiltinReshape:
            return VisitReshapeOperator(delegateData,
                                        tfLiteContext,
                                        tfLiteNode,
                                        nodeIndex,
                                        kTfLiteBuiltinReshape);
        case kTfLiteBuiltinResizeBilinear:
            return VisitResizeOperator(delegateData,
                                       tfLiteContext,
                                       tfLiteNode,
                                       nodeIndex,
                                       kTfLiteBuiltinResizeBilinear);
        case kTfLiteBuiltinResizeNearestNeighbor:
            return VisitResizeOperator(delegateData,
                                       tfLiteContext,
                                       tfLiteNode,
                                       nodeIndex,
                                       kTfLiteBuiltinResizeNearestNeighbor);
        case kTfLiteBuiltinRsqrt:
            return VisitElementwiseUnaryOperator(delegateData,
                                                 tfLiteContext,
                                                 tfLiteNode,
                                                 nodeIndex,
                                                 armnn::UnaryOperation::Rsqrt);
        case kTfLiteBuiltinShape:
            return VisitShapeOperator(delegateData,
                                      tfLiteContext,
                                      tfLiteNode,
                                      nodeIndex,
                                      kTfLiteBuiltinShape);
        case kTfLiteBuiltinSplit:
            return VisitSplitOperator(delegateData,
                                      tfLiteContext,
                                      tfLiteNode,
                                      nodeIndex,
                                      kTfLiteBuiltinSplit);
        case kTfLiteBuiltinSplitV:
            return VisitSplitVOperator(delegateData,
                                       tfLiteContext,
                                       tfLiteNode,
                                       nodeIndex,
                                       kTfLiteBuiltinSplitV);
        case kTfLiteBuiltinSqrt:
            return VisitElementwiseUnaryOperator(delegateData,
                                                 tfLiteContext,
                                                 tfLiteNode,
                                                 nodeIndex,
                                                 armnn::UnaryOperation::Sqrt);
        case kTfLiteBuiltinSqueeze:
            return VisitSqueezeOperator(delegateData,
                                        tfLiteContext,
                                        tfLiteNode,
                                        nodeIndex,
                                        kTfLiteBuiltinSqueeze);
        case kTfLiteBuiltinStridedSlice:
            return VisitSliceOperator(delegateData,
                                      tfLiteContext,
                                      tfLiteNode,
                                      nodeIndex,
                                      kTfLiteBuiltinStridedSlice);
        case kTfLiteBuiltinSum:
            return VisitReduceOperator(delegateData,
                                       tfLiteContext,
                                       tfLiteNode,
                                       nodeIndex,
                                       kTfLiteBuiltinSum);
        case kTfLiteBuiltinTranspose:
            return VisitTransposeOperator(delegateData,
                                          tfLiteContext,
                                          tfLiteNode,
                                          nodeIndex,
                                          kTfLiteBuiltinTranspose);
        case kTfLiteBuiltinTransposeConv:
            return VisitConvolutionOperator(delegateData,
                                            tfLiteContext,
                                            tfLiteNode,
                                            nodeIndex,
                                            kTfLiteBuiltinTransposeConv);
        case kTfLiteBuiltinSoftmax:
            return VisitSoftmaxOperator(delegateData,
                                        tfLiteContext,
                                        tfLiteNode,
                                        nodeIndex,
                                        kTfLiteBuiltinSoftmax);
        case kTfLiteBuiltinSpaceToBatchNd:
            return VisitSpaceToBatchNdOperator(delegateData,
                                               tfLiteContext,
                                               tfLiteNode,
                                               nodeIndex,
                                               kTfLiteBuiltinSpaceToBatchNd);
        case kTfLiteBuiltinSpaceToDepth:
            return VisitSpaceToDepthOperator(delegateData,
                                             tfLiteContext,
                                             tfLiteNode,
                                             nodeIndex,
                                             kTfLiteBuiltinSpaceToDepth);
        case kTfLiteBuiltinSub:
            return VisitElementwiseBinaryOperator(delegateData,
                                                  tfLiteContext,
                                                  tfLiteNode,
                                                  nodeIndex,
                                                  kTfLiteBuiltinSub);
        case kTfLiteBuiltinTanh:
            return VisitActivationOperator(delegateData,
                                           tfLiteContext,
                                           tfLiteNode,
                                           nodeIndex,
                                           kTfLiteBuiltinTanh);
        case kTfLiteBuiltinUnidirectionalSequenceLstm:
            return VisitUnidirectionalSequenceLstmOperator(delegateData,
                                                           tfLiteContext,
                                                           tfLiteNode,
                                                           nodeIndex,
                                                           kTfLiteBuiltinUnidirectionalSequenceLstm);
        case kTfLiteBuiltinUnpack:
            return VisitUnpackOperator(delegateData,
                                       tfLiteContext,
                                       tfLiteNode,
                                       nodeIndex,
                                       kTfLiteBuiltinUnpack);
        default:
            return kTfLiteError;
    }
}

} // armnnDelegate namespace