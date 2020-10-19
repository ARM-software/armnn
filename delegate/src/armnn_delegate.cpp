//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn_delegate.hpp>
#include <algorithm>

namespace armnnDelegate
{

Delegate::Delegate(armnnDelegate::DelegateOptions options)
  : m_Runtime(nullptr, nullptr),
    m_Options(std::move(options))
{
    // Create ArmNN Runtime
    armnn::IRuntime::CreationOptions runtimeOptions;
    m_Runtime = armnn::IRuntime::Create(runtimeOptions);

    std::vector<armnn::BackendId> backends;

    if (m_Runtime)
    {
        const armnn::BackendIdSet supportedDevices = m_Runtime->GetDeviceSpec().GetSupportedBackends();
        for (auto& backend : m_Options.GetBackends())
        {
            if (std::find(supportedDevices.cbegin(), supportedDevices.cend(), backend) == supportedDevices.cend())
            {
                TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
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

TfLiteIntArray* Delegate::CollectOperatorsToDelegate(TfLiteContext* tfLiteContext)
{
    TfLiteIntArray* executionPlan = nullptr;
    if (tfLiteContext->GetExecutionPlan(tfLiteContext, &executionPlan) != kTfLiteOk)
    {
        TF_LITE_KERNEL_LOG(tfLiteContext, "TfLiteArmnnDelegate: Unable to get graph execution plan.");
        return nullptr;
    }

    // Null INetworkPtr
    armnn::INetworkPtr nullNetworkPtr(nullptr, nullptr);

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
            nullNetworkPtr, tfLiteContext, tfLiteRegistration, tfLiteNode, nodeIndex) != kTfLiteOk)
        {
            // node is not supported by ArmNN
            continue;
        }

        nodesToDelegate->data[nodesToDelegate->size++] = nodeIndex;
    }

    std::sort(&nodesToDelegate->data[0],
              &nodesToDelegate->data[nodesToDelegate->size]);

    return nodesToDelegate;
}

TfLiteDelegate* Delegate::GetDelegate()
{
    return &m_Delegate;
}

ArmnnSubgraph* ArmnnSubgraph::Create(TfLiteContext* tfLiteContext,
                                     const TfLiteDelegateParams* parameters,
                                     const Delegate* delegate)
{
    TfLiteIntArray* executionPlan;
    if (tfLiteContext->GetExecutionPlan(tfLiteContext, &executionPlan) != kTfLiteOk)
    {
        return nullptr;
    }

    // Construct ArmNN network
    using NetworkOptions = std::vector<armnn::BackendOptions>;
    armnn::NetworkOptions networkOptions = {};
    armnn::NetworkId networkId;
    armnn::INetworkPtr network = armnn::INetwork::Create(networkOptions);

    // Parse TfLite delegate nodes to ArmNN nodes
    for (int i = 0; i < parameters->nodes_to_replace->size; ++i)
    {
        const int nodeIndex = parameters->nodes_to_replace->data[i];

        TfLiteNode* tfLiteNode = nullptr;
        TfLiteRegistration* tfLiteRegistration = nullptr;
        if (tfLiteContext->GetNodeAndRegistration(
            tfLiteContext, nodeIndex, &tfLiteNode, &tfLiteRegistration) != kTfLiteOk)
        {
            throw armnn::Exception("TfLiteArmnnDelegate: Unable to get node registration: " + nodeIndex);
        }

        if (VisitNode(network, tfLiteContext, tfLiteRegistration, tfLiteNode, nodeIndex) != kTfLiteOk)
        {
            throw armnn::Exception("TfLiteArmnnDelegate: Unable to parse node: " + nodeIndex);
        }
    }

    // Optimise Arm NN network
    armnn::IOptimizedNetworkPtr optNet =
        armnn::Optimize(*network, delegate->m_Options.GetBackends(), delegate->m_Runtime->GetDeviceSpec());
    if (!optNet)
    {
        // Optimize Failed
        throw armnn::Exception("TfLiteArmnnDelegate: Unable to optimize the network!");
    }
    // Load graph into runtime
    delegate->m_Runtime->LoadNetwork(networkId, std::move(optNet));

    // Create a new SubGraph with networkId and runtime
    return new ArmnnSubgraph(networkId, delegate->m_Runtime.get());
}

TfLiteStatus ArmnnSubgraph::Prepare(TfLiteContext* tfLiteContext)
{
    return kTfLiteOk;
}

TfLiteStatus ArmnnSubgraph::Invoke(TfLiteContext* tfLiteContext)
{
    /// Get the Input Tensors and OutputTensors from the context
    /// Execute the network
    //m_Runtime->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

    return kTfLiteOk;
}

TfLiteStatus ArmnnSubgraph::VisitNode(armnn::INetworkPtr& network,
                                      TfLiteContext* tfLiteContext,
                                      TfLiteRegistration* tfLiteRegistration,
                                      TfLiteNode* tfLiteNode,
                                      int nodeIndex)
{
    /*
     * Take the node and check what operator it is and VisitXXXLayer()
     * In the VisitXXXLayer() function parse TfLite node to Arm NN Layer and add it to tho network graph
     *switch (tfLiteRegistration->builtin_code)
     * {
     *     case kTfLiteBuiltinAbs:
     *              return VisitAbsLayer(...);
     *      ...
     *      default:
     *          return kTfLiteError;
     *  }
     */
    return kTfLiteError;
}

} // armnnDelegate namespace