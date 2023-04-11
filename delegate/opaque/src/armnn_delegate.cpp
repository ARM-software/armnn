//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn_delegate.hpp>
#include <OpaqueDelegateUtils.hpp>

#include <Version.hpp>

#include "Activation.hpp"
#include "ArgMinMax.hpp"
#include "BatchMatMul.hpp"
#include "BatchSpace.hpp"
#include "Comparison.hpp"
#include "Convolution.hpp"
#include "Control.hpp"
#include "ElementwiseBinary.hpp"
#include "ElementwiseUnary.hpp"
#include "Fill.hpp"
#include "FullyConnected.hpp"
#include "Gather.hpp"
#include "GatherNd.hpp"
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
#include "StridedSlice.hpp"
#include "Softmax.hpp"
#include "SpaceDepth.hpp"
#include "Split.hpp"
#include "Transpose.hpp"
#include "UnidirectionalSequenceLstm.hpp"
#include "Unpack.hpp"

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnnUtils/Filesystem.hpp>
#include <armnn/utility/Timer.hpp>
#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/context_util.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/minimal_logging.h>
#include <tensorflow/lite/logger.h>

#include <algorithm>
#include <iostream>
#include <sstream>

namespace armnnOpaqueDelegate
{

const TfLiteStableDelegate TFL_TheStableDelegate =
{
    /*delegate_abi_version=*/ TFL_STABLE_DELEGATE_ABI_VERSION,
    /*delegate_name=*/        "ArmnnDelegatePlugin",
    /*delegate_version=*/     "1.0.0",
    /*delegate_plugin=*/      GetArmnnDelegatePluginApi()
};

ArmnnOpaqueDelegate::ArmnnOpaqueDelegate(armnnDelegate::DelegateOptions options)
    : m_Options(std::move(options))
{
    // Configures logging for ARMNN
    if (m_Options.IsLoggingEnabled())
    {
        armnn::ConfigureLogging(true, true, m_Options.GetLoggingSeverity());
    }
    // Create/Get the static ArmNN Runtime. Note that the m_Runtime will be shared by all armnn_delegate
    // instances so the RuntimeOptions cannot be altered for different armnn_delegate instances.
    m_Runtime = GetRuntime(m_Options.GetRuntimeOptions());
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
        throw armnn::InvalidArgumentException("TfLiteArmnnOpaqueDelegate: No known backend specified.");
    }
    m_Options.SetBackends(backends);

    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO, "TfLiteArmnnOpaqueDelegate: Created TfLite ArmNN delegate.");
}

TfLiteStatus DoPrepare(TfLiteOpaqueContext* tfLiteContext, TfLiteOpaqueDelegate* tfLiteDelegate, void* data)
{
    // We are required to have the void* data parameter in the function signature, but we don't actually use it.
    armnn::IgnoreUnused(data);

    TfLiteIntArray* supportedOperators =
            static_cast<::armnnOpaqueDelegate::ArmnnOpaqueDelegate*>
                    (TfLiteOpaqueDelegateGetData(tfLiteDelegate))->IdentifyOperatorsToDelegate(tfLiteContext);
    if(supportedOperators == nullptr)
    {
        return kTfLiteError;
    }

    // ArmNN Opaque Delegate Registration
    TfLiteRegistrationExternal* kernelRegistration =
            TfLiteRegistrationExternalCreate(kTfLiteBuiltinDelegate, "TfLiteArmNNOpaqueDelegate", /*version=*/1);
    if(kernelRegistration == nullptr)
    {
        return kTfLiteError;
    }

    TfLiteRegistrationExternalSetInit(
            kernelRegistration,
            [](TfLiteOpaqueContext* tfLiteContext, const char* buffer, size_t length) -> void*
            {
                armnn::IgnoreUnused(length);
                const TfLiteOpaqueDelegateParams* parameters =
                        reinterpret_cast<const TfLiteOpaqueDelegateParams*>(buffer);
                if(parameters == nullptr)
                {
                    TF_LITE_OPAQUE_KERNEL_LOG(tfLiteContext,
                                              "TfLiteArmnnOpaqueDelegate: Unable to get parameters.");
                    return nullptr;
                }

                return static_cast<void*>(
                        ArmnnSubgraph::Create(tfLiteContext,
                                              parameters,
                                              static_cast<::armnnOpaqueDelegate::ArmnnOpaqueDelegate*>(
                                                      parameters->delegate->opaque_delegate_builder->data)));
            }
    );

    TfLiteRegistrationExternalSetFree(
            kernelRegistration,
            [](TfLiteOpaqueContext* tfLiteContext, void* buffer) -> void
            {
                armnn::IgnoreUnused(tfLiteContext);
                if (buffer != nullptr)
                {
                    delete static_cast<ArmnnSubgraph*>(buffer);
                }
            }
    );

    TfLiteRegistrationExternalSetPrepare(
            kernelRegistration,
            [](TfLiteOpaqueContext* tfLiteContext, TfLiteOpaqueNode* tfLiteNode) -> TfLiteStatus
            {
                void* userData = TfLiteOpaqueNodeGetUserData(tfLiteNode);
                if (userData == nullptr)
                {
                    return kTfLiteError;
                }
                return static_cast<ArmnnSubgraph*>(userData)->Prepare(tfLiteContext);
            }
    );

    TfLiteRegistrationExternalSetInvoke(
            kernelRegistration,
            [](TfLiteOpaqueContext* tfLiteContext, TfLiteOpaqueNode* tfLiteNode) -> TfLiteStatus
            {
                void* userData = TfLiteOpaqueNodeGetUserData(tfLiteNode);
                if (userData == nullptr)
                {
                    return kTfLiteError;
                }

                return static_cast<ArmnnSubgraph*>(userData)->Invoke(tfLiteContext, tfLiteNode);
            }
    );

    const TfLiteStatus status =
            TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
                    tfLiteContext, kernelRegistration, supportedOperators, tfLiteDelegate);

    TfLiteIntArrayFree(supportedOperators);
    return status;
}

TfLiteOpaqueDelegate* TfLiteArmnnOpaqueDelegateCreate(const void* settings)
{
    // This method will always create Opaque Delegate with default settings until
    // we have a DelegateOptions Constructor which can parse the void* settings
    armnn::IgnoreUnused(settings);
    auto options = TfLiteArmnnDelegateOptionsDefault();
    auto* armnnDelegate = new ::armnnOpaqueDelegate::ArmnnOpaqueDelegate(options);
    return TfLiteOpaqueDelegateCreate(armnnDelegate->GetDelegateBuilder());
}

::armnnDelegate::DelegateOptions TfLiteArmnnDelegateOptionsDefault()
{
    ::armnnDelegate::DelegateOptions options(armnn::Compute::CpuRef);
    return options;
}

void TfLiteArmnnOpaqueDelegateDelete(TfLiteOpaqueDelegate* tfLiteDelegate)
{
    if (tfLiteDelegate != nullptr)
    {
        delete static_cast<::armnnOpaqueDelegate::ArmnnOpaqueDelegate*>(TfLiteOpaqueDelegateGetData(tfLiteDelegate));
        TfLiteOpaqueDelegateDelete(tfLiteDelegate);
    }
}

const TfLiteOpaqueDelegatePlugin* GetArmnnDelegatePluginApi()
{
    static constexpr TfLiteOpaqueDelegatePlugin armnnPlugin{
            TfLiteArmnnOpaqueDelegateCreate, TfLiteArmnnOpaqueDelegateDelete, TfLiteArmnnOpaqueDelegateErrno};
    return &armnnPlugin;
}

const std::string ArmnnOpaqueDelegate::GetVersion() {
    return OPAQUE_DELEGATE_VERSION;
}

TfLiteIntArray* ArmnnOpaqueDelegate::IdentifyOperatorsToDelegate(TfLiteOpaqueContext* tfLiteContext)
{
    TfLiteIntArray* executionPlan = nullptr;
    if (TfLiteOpaqueContextGetExecutionPlan(tfLiteContext, &executionPlan) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_KERNEL_LOG(tfLiteContext, "TfLiteArmnnOpaqueDelegate: Unable to get graph execution plan.");
        return nullptr;
    }

    // Delegate data with null network
    DelegateData delegateData(m_Options.GetBackends());

    TfLiteIntArray* nodesToDelegate = TfLiteIntArrayCreate(executionPlan->size);
    if (nodesToDelegate == nullptr)
    {
        TF_LITE_OPAQUE_KERNEL_LOG(tfLiteContext,
                                  "TfLiteArmnnOpaqueDelegate: Unable to create int array from execution plan.");
        return nullptr;
    }
    nodesToDelegate->size = 0;

    std::set<int32_t> unsupportedOperators;

    for (int i = 0; i < executionPlan->size; ++i)
    {
        const int nodeIndex = executionPlan->data[i];

        // If TfLiteOpaqueNodes can be delegated to ArmNN
        TfLiteOpaqueNode* tfLiteNode = nullptr;
        TfLiteRegistrationExternal* tfLiteRegistration = nullptr;

        if (TfLiteOpaqueContextGetNodeAndRegistration(
                tfLiteContext, nodeIndex, &tfLiteNode, &tfLiteRegistration) != kTfLiteOk)
        {
            TF_LITE_OPAQUE_KERNEL_LOG(tfLiteContext,
                                      "TfLiteArmnnOpaqueDelegate: Unable to get node and registration for node %d.",
                                      nodeIndex);
            continue;
        }

        TfLiteStatus visitStatus;
        try
        {
            visitStatus = ArmnnSubgraph::VisitNode(
                    delegateData, tfLiteContext, tfLiteRegistration, tfLiteNode, nodeIndex);
        }
        catch(std::exception& ex)
        {
            ARMNN_LOG(error) << "ArmNN Failed to visit node with error: " << ex.what();
            visitStatus = kTfLiteError;
        }

        if (visitStatus != kTfLiteOk)
        {
            // node is not supported by ArmNN
            unsupportedOperators.insert(TfLiteRegistrationExternalGetBuiltInCode(tfLiteRegistration));
            continue;
        }

        nodesToDelegate->data[nodesToDelegate->size++] = nodeIndex;
    }

    for (std::set<int32_t>::iterator it=unsupportedOperators.begin(); it!=unsupportedOperators.end(); ++it)
    {
        TF_LITE_OPAQUE_KERNEL_LOG(tfLiteContext,
                                  "Operator %s [%d] is not supported by armnn_opaque_delegate.",
                                  tflite::EnumNameBuiltinOperator(tflite::BuiltinOperator(*it)),
                                  *it);
    }

    if (!unsupportedOperators.empty() && m_Options.TfLiteRuntimeFallbackDisabled())
    {
        std::stringstream exMessage;
        exMessage << "TfLiteArmnnOpaqueDelegate: There are unsupported operators in the model. ";
        exMessage << "Not falling back to TfLite Runtime as fallback is disabled. ";
        exMessage << "This should only be disabled under test conditions.";
        throw armnn::Exception(exMessage.str());
    }
    if (nodesToDelegate->size == 0)
    {
        ARMNN_LOG(info) << "No operators in this model are supported by the Arm NN TfLite delegate." <<
                        " The model will be executed entirely by TfLite runtime.";
    }

    std::sort(&nodesToDelegate->data[0], &nodesToDelegate->data[nodesToDelegate->size]);
    return nodesToDelegate;
}

TfLiteStatus ArmnnSubgraph::AddInputLayer(DelegateData& delegateData,
                                          TfLiteOpaqueContext* tfLiteContext,
                                          const TfLiteIntArray* inputs,
                                          std::vector<armnn::BindingPointInfo>& inputBindings)
{
    const size_t numInputs = static_cast<size_t>(inputs->size);
    for (unsigned int i = 0; i < numInputs; ++i)
    {
        const int32_t tensorId = inputs->data[i];
        const TfLiteOpaqueTensor* tensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, tensorId);

        if(!tensor)
        {
            return kTfLiteError;
        }

        // Do not create bindings for constant inputs
        if (TfLiteOpaqueTensorGetAllocationType(tensor) == kTfLiteMmapRo)
        {
            continue;
        }

        auto bindingId = static_cast<armnn::LayerBindingId>((tensorId));
        armnn::IConnectableLayer* layer = delegateData.m_Network->AddInputLayer(bindingId);

        auto tensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tensor);
        armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
        outputSlot.SetTensorInfo(tensorInfo);

        // Store for creating connections
        delegateData.m_OutputSlotForNode[static_cast<unsigned long>(tensorId)] = &outputSlot;

        inputBindings.push_back(std::make_pair(bindingId, tensorInfo));
    }

    return kTfLiteOk;
}

TfLiteStatus ArmnnSubgraph::AddOutputLayer(DelegateData& delegateData,
                                           TfLiteOpaqueContext* tfLiteContext,
                                           const TfLiteIntArray* outputs,
                                           std::vector<armnn::BindingPointInfo>& outputBindings)
{
    const size_t numOutputs = static_cast<size_t>(outputs->size);
    for (unsigned int i = 0; i < numOutputs; ++i)
    {
        const int32_t tensorId = outputs->data[i];
        const TfLiteOpaqueTensor* tensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, tensorId);

        if(!IsValid(tensor))
        {
            return kTfLiteError;
        }

        auto bindingId = static_cast<armnn::LayerBindingId>((tensorId));
        armnn::IConnectableLayer* layer = delegateData.m_Network->AddOutputLayer(bindingId);

        auto tensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tensor);
        ARMNN_ASSERT(delegateData.m_OutputSlotForNode[static_cast<unsigned long>(tensorId)] != nullptr);
        delegateData.m_OutputSlotForNode[static_cast<unsigned long>(tensorId)]->Connect(layer->GetInputSlot(0));
        outputBindings.push_back(std::make_pair(bindingId, tensorInfo));
    }

    return kTfLiteOk;
}

ArmnnSubgraph* ArmnnSubgraph::Create(TfLiteOpaqueContext* tfLiteContext,
                                     const TfLiteOpaqueDelegateParams* parameters,
                                     const ArmnnOpaqueDelegate* delegate)
{
    const auto startTime = armnn::GetTimeNow();
    ARMNN_LOG(info) << "ArmnnSubgraph creation";

    TfLiteIntArray* executionPlan;
    if (TfLiteOpaqueContextGetExecutionPlan(tfLiteContext, &executionPlan) != kTfLiteOk)
    {
        return nullptr;
    }

    // Initialize DelegateData holds network and output slots information
    DelegateData delegateData(delegate->m_Options.GetBackends());

    // Build ArmNN Network
    armnn::NetworkOptions networkOptions = delegate->m_Options.GetOptimizerOptions().GetModelOptions();
    armnn::NetworkId networkId;
    delegateData.m_Network = armnn::INetwork::Create(networkOptions);

    delegateData.m_OutputSlotForNode = std::vector<armnn::IOutputSlot*>(
                                                            TfLiteOpaqueContextGetNumTensors(tfLiteContext), nullptr);

    std::vector<armnn::BindingPointInfo> inputBindings;
    std::vector<armnn::BindingPointInfo> outputBindings;

    // Add input layer
    if (AddInputLayer(delegateData, tfLiteContext, parameters->input_tensors, inputBindings) != kTfLiteOk)
    {
        throw armnn::Exception("TfLiteArmnnOpaqueDelegate: Unable to add Inputs to the network!");
    }

    // Parse TfLite delegate nodes to ArmNN
    const auto parseStartTime = armnn::GetTimeNow();
    for (int i = 0; i < parameters->nodes_to_replace->size; ++i)
    {
        const int nodeIndex = parameters->nodes_to_replace->data[i];

        TfLiteOpaqueNode* tfLiteNode = nullptr;
        TfLiteRegistrationExternal* tfLiteRegistration = nullptr;
        if (TfLiteOpaqueContextGetNodeAndRegistration(
            tfLiteContext, nodeIndex, &tfLiteNode, &tfLiteRegistration) != kTfLiteOk)
        {
            throw armnn::Exception(&"TfLiteArmnnOpaqueDelegate: Unable to get node registration: " [ nodeIndex]);
        }

        if (VisitNode(delegateData, tfLiteContext, tfLiteRegistration, tfLiteNode, nodeIndex) != kTfLiteOk)
        {
            throw armnn::Exception(&"TfLiteArmnnOpaqueDelegate: Unable to parse node: " [ nodeIndex]);
        }
    }
    ARMNN_LOG(info) << "Parse nodes to ArmNN time: " << std::setprecision(2)
                    << std::fixed << armnn::GetTimeDuration(parseStartTime).count() << " ms";

    // Add Output layer
    if (AddOutputLayer(delegateData, tfLiteContext, parameters->output_tensors, outputBindings) != kTfLiteOk)
    {
        throw armnn::Exception("TfLiteArmnnOpaqueDelegate: Unable to add Outputs to the network!");
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
    catch (std::exception& ex)
    {
        std::stringstream exMessage;
        exMessage << "TfLiteArmnnOpaqueDelegate: Exception (" << ex.what() << ") caught from optimize.";
        throw armnn::Exception(exMessage.str());
    }
    if (!optNet)
    {
        // Optimize failed
        throw armnn::Exception("TfLiteArmnnOpaqueDelegate: Unable to optimize the network!");
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
        armnn::MemorySource inputSource = armnn::MemorySource::Undefined;
        armnn::MemorySource outputSource = armnn::MemorySource::Undefined;
        // There's a bit of an assumption here that the delegate will only support Malloc memory source.
        if (delegate->m_Options.GetOptimizerOptions().GetImportEnabled())
        {
            inputSource = armnn::MemorySource::Malloc;
        }
        if (delegate->m_Options.GetOptimizerOptions().GetExportEnabled())
        {
            outputSource = armnn::MemorySource::Malloc;
        }
        armnn::INetworkProperties networkProperties(false,
                                                    inputSource,
                                                    outputSource,
                                                    delegate->m_Options.GetInternalProfilingState(),
                                                    delegate->m_Options.GetInternalProfilingDetail());
        loadingStatus = delegate->m_Runtime->LoadNetwork(networkId,
                                                         std::move(optNet),
                                                         errorMessage,
                                                         networkProperties);
        if (loadingStatus != armnn::Status::Success)
        {
            // Network load failed.
            throw armnn::Exception("TfLiteArmnnOpaqueDelegate: Network could not be loaded: " + errorMessage);
        }

        ARMNN_LOG(info) << "Load ArmnnSubgraph time: " << std::setprecision(2)
                        << std::fixed << armnn::GetTimeDuration(loadStartTime).count() << " ms";
    }
    catch (std::exception& ex)
    {
        std::stringstream exMessage;
        exMessage << "TfLiteArmnnOpaqueDelegate: Exception (" << ex.what() << ") caught from LoadNetwork.";
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
    return new ArmnnSubgraph(networkId, delegate->m_Runtime, inputBindings, outputBindings);
}

TfLiteStatus ArmnnSubgraph::Prepare(TfLiteOpaqueContext* tfLiteContext)
{
    armnn::IgnoreUnused(tfLiteContext);
    return kTfLiteOk;
}

TfLiteStatus ArmnnSubgraph::Invoke(TfLiteOpaqueContext* tfLiteContext, TfLiteOpaqueNode* tfLiteNode)
{
    // Get array of input indices, inputIndexArray is set from the TfLiteOpaqueNodeInputs function
    // This function turns inputIndexArray into an int array of indices. These indices point to the tensors for
    // each input slot in the node.
    const int* inputIndexArray;
    int numInputs;
    if(TfLiteOpaqueNodeInputs(tfLiteNode, &inputIndexArray, &numInputs) != kTfLiteOk)
    {
        throw armnn::Exception("TfLiteArmnnOpaqueDelegate: Unable to load subgraph inputs!");
    }
    // Prepare inputs
    armnn::InputTensors inputTensors;
    size_t inputIndex = 0;
    for (int inputIdx = 0; inputIdx < numInputs; inputIdx++)
    {
        TfLiteOpaqueTensor* tensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputIndexArray[inputIdx]);

        if(!IsValid(tensor))
        {
            return kTfLiteError;
        }
        // If tensor is not read only
        if (TfLiteOpaqueTensorGetAllocationType(tensor) != kTfLiteMmapRo)
        {
            const armnn::BindingPointInfo& inputBinding = m_InputBindings[inputIndex];
            armnn::TensorInfo inputTensorInfo = inputBinding.second;
            inputTensorInfo.SetConstant(true);
            const armnn::ConstTensor inputTensor(inputTensorInfo, TfLiteOpaqueTensorData(tensor));
            inputTensors.emplace_back(inputIdx, inputTensor);

            ++inputIndex;
        }
    }

    // Get array of output indices, outputIndexArray is set from the TfLiteOpaqueNodeOutputs function
    // This function turns outputIndexArray into an int array of indices. These indices point to the tensors for
    // each output slot in the node.
    const int* outputIndexArray;
    int numOutputs;
    if(TfLiteOpaqueNodeOutputs(tfLiteNode, &outputIndexArray, &numOutputs) != kTfLiteOk)
    {
        throw armnn::Exception("TfLiteArmnnOpaqueDelegate: Unable to load subgraph outputs!");
    }
    // Assign the tensors from the outputIndexArray to the armnn BindingPointInfo
    armnn::OutputTensors outputTensors;
    for (int outputIdx = 0; outputIdx < numOutputs; outputIdx++)
    {
        const armnn::BindingPointInfo& outputBinding = m_OutputBindings[outputIdx];
        TfLiteOpaqueTensor* tensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputIndexArray[outputIdx]);
        if(!IsValid(tensor))
        {
            return kTfLiteError;
        }

        const armnn::Tensor outputTensor(outputBinding.second, reinterpret_cast<TfLiteTensor*>(tensor)->data
        .data);
        outputTensors.emplace_back(outputIndexArray[outputIdx], outputTensor);
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
                                      TfLiteOpaqueContext* tfLiteContext,
                                      TfLiteRegistrationExternal* tfLiteRegistration,
                                      TfLiteOpaqueNode* tfLiteNode,
                                      int nodeIndex)
{
    switch (TfLiteRegistrationExternalGetBuiltInCode(tfLiteRegistration))
    {
        case kTfLiteBuiltinCast:
            return VisitCastOperator(delegateData,
                                     tfLiteContext,
                                     tfLiteNode,
                                     nodeIndex,
                                     kTfLiteBuiltinCast);
        default:
            return kTfLiteError;
    }
}
} // armnnOpaqueDelegate namespace