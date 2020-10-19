//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DelegateOptions.hpp"

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnDelegate
{

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);

/// Delegate class
class Delegate
{
    friend class ArmnnSubgraph;
public:
    explicit Delegate(armnnDelegate::DelegateOptions options);

    TfLiteIntArray* CollectOperatorsToDelegate(TfLiteContext* context);

    TfLiteDelegate* GetDelegate();

private:
    TfLiteDelegate m_Delegate = {
        reinterpret_cast<void*>(this),  // .data_
        DelegatePrepare,                // .Prepare
        nullptr,                        // .CopyFromBufferHandle
        nullptr,                        // .CopyToBufferHandle
        nullptr,                        // .FreeBufferHandle
        kTfLiteDelegateFlagsNone,       // .flags
    };

    /// Arm NN Runtime pointer
    armnn::IRuntimePtr m_Runtime;
    /// Arm NN Delegate Options
    armnnDelegate::DelegateOptions m_Options;
};

/// ArmnnSubgraph class where parsing the nodes to ArmNN format and creating the ArmNN Graph
class ArmnnSubgraph
{
public:
    static ArmnnSubgraph* Create(TfLiteContext* tfLiteContext,
                                 const TfLiteDelegateParams* parameters,
                                 const Delegate* delegate);

    TfLiteStatus Prepare(TfLiteContext* tfLiteContext);

    TfLiteStatus Invoke(TfLiteContext* tfLiteContext);

    static TfLiteStatus VisitNode(armnn::INetworkPtr& network,
                                  TfLiteContext* tfLiteContext,
                                  TfLiteRegistration* tfLiteRegistration,
                                  TfLiteNode* tfLiteNode,
                                  int nodeIndex);

private:
    ArmnnSubgraph(armnn::NetworkId networkId, armnn::IRuntime* runtime)
        : m_NetworkId(networkId), m_Runtime(runtime)
    {}

    /// The Network Id
    armnn::NetworkId m_NetworkId;
    /// ArmNN Rumtime
    armnn::IRuntime* m_Runtime;
};

void* ArmnnSubgraphInit(TfLiteContext* tfLiteContext, const char* buffer, size_t length)
{
    const TfLiteDelegateParams* parameters = reinterpret_cast<const TfLiteDelegateParams*>(buffer);

    return static_cast<void*>(ArmnnSubgraph::Create(
        tfLiteContext, parameters, static_cast<::armnnDelegate::Delegate*>(parameters->delegate->data_)));
}

TfLiteStatus ArmnnSubgraphPrepare(TfLiteContext* tfLiteContext, TfLiteNode* tfLiteNode)
{
    if (tfLiteNode->user_data == nullptr)
    {
        return kTfLiteError;
    }

    return static_cast<ArmnnSubgraph*>(tfLiteNode->user_data)->Prepare(tfLiteContext);
}

TfLiteStatus ArmnnSubgraphInvoke(TfLiteContext* tfLiteContext, TfLiteNode* tfLiteNode)
{
    if (tfLiteNode->user_data == nullptr)
    {
        return kTfLiteError;
    }

    return static_cast<ArmnnSubgraph*>(tfLiteNode->user_data)->Invoke(tfLiteContext);
}

void ArmnnSubgraphFree(TfLiteContext* tfLiteContext, void* buffer)
{
    if (buffer != nullptr)
    {
        delete static_cast<ArmnnSubgraph*>(buffer);
    }
}

const TfLiteRegistration armnnSubgraphRegistration = {
    ArmnnSubgraphInit,     // .init
    ArmnnSubgraphFree,     // .free
    ArmnnSubgraphPrepare,  // .prepare
    ArmnnSubgraphInvoke,   // .invoke
    nullptr,               // .profiling_string
    0,                     // .builtin_code
    "TfLiteArmnnDelegate", // .custom_name
    1,                     // .version
};

TfLiteStatus DelegatePrepare(TfLiteContext* tfLiteContext, TfLiteDelegate* tfLiteDelegate)
{
    TfLiteIntArray* supportedOperators =
        static_cast<::armnnDelegate::Delegate*>(tfLiteDelegate->data_)->CollectOperatorsToDelegate(tfLiteContext);

    const TfLiteStatus status =
        tfLiteContext->ReplaceNodeSubsetsWithDelegateKernels(
            tfLiteContext, armnnSubgraphRegistration, supportedOperators, tfLiteDelegate);
    TfLiteIntArrayFree(supportedOperators);

    return status;
}

} // armnnDelegate namespace

armnnDelegate::DelegateOptions TfLiteArmnnDelegateOptionsDefault() {
    armnnDelegate::DelegateOptions options(armnn::Compute::CpuRef);
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