//
// Copyright © 2020-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <DelegateOptions.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include <tensorflow/lite/version.h>

#if TF_MAJOR_VERSION > 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION > 3)
#define ARMNN_POST_TFLITE_2_3
#endif

#if TF_MAJOR_VERSION > 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION > 4)
#define ARMNN_POST_TFLITE_2_4
#endif

#if TF_MAJOR_VERSION > 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION > 5)
#define ARMNN_POST_TFLITE_2_5
#endif

namespace armnnDelegate
{

struct DelegateData
{
    DelegateData(const std::vector<armnn::BackendId>& backends)
            : m_Backends(backends)
            , m_Network(nullptr, nullptr)
    {}

    const std::vector<armnn::BackendId>       m_Backends;
    armnn::INetworkPtr                        m_Network;
    std::vector<armnn::IOutputSlot*>          m_OutputSlotForNode;
};

// Forward decleration for functions initializing the ArmNN Delegate
DelegateOptions TfLiteArmnnDelegateOptionsDefault();

TfLiteDelegate* TfLiteArmnnDelegateCreate(armnnDelegate::DelegateOptions options);

void TfLiteArmnnDelegateDelete(TfLiteDelegate* tfLiteDelegate);

TfLiteStatus DoPrepare(TfLiteContext* context, TfLiteDelegate* delegate);

/// ArmNN Delegate
class Delegate
{
    friend class ArmnnSubgraph;
public:
    explicit Delegate(armnnDelegate::DelegateOptions options);

    TfLiteIntArray* IdentifyOperatorsToDelegate(TfLiteContext* context);

    TfLiteDelegate* GetDelegate();

    /// Retrieve version in X.Y.Z form
    static const std::string GetVersion();

private:
    /**
     * Returns a pointer to the armnn::IRuntime* this will be shared by all armnn_delegates.
     */
    armnn::IRuntime* GetRuntime(const armnn::IRuntime::CreationOptions& options)
    {
        static armnn::IRuntimePtr instance = armnn::IRuntime::Create(options);
        // Instantiated on first use.
        return instance.get();
    }

    TfLiteDelegate m_Delegate = {
            reinterpret_cast<void*>(this),  // .data_
            DoPrepare,                      // .Prepare
            nullptr,                        // .CopyFromBufferHandle
            nullptr,                        // .CopyToBufferHandle
            nullptr,                        // .FreeBufferHandle
            kTfLiteDelegateFlagsNone,       // .flags
            nullptr,                        // .opaque_delegate_builder
    };

    /// ArmNN Runtime pointer
    armnn::IRuntime* m_Runtime;
    /// ArmNN Delegate Options
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

    TfLiteStatus Invoke(TfLiteContext* tfLiteContext, TfLiteNode* tfLiteNode);

    static TfLiteStatus VisitNode(DelegateData& delegateData,
                                  TfLiteContext* tfLiteContext,
                                  TfLiteRegistration* tfLiteRegistration,
                                  TfLiteNode* tfLiteNode,
                                  int nodeIndex);

private:
    ArmnnSubgraph(armnn::NetworkId networkId,
                  armnn::IRuntime* runtime,
                  std::vector<armnn::BindingPointInfo>& inputBindings,
                  std::vector<armnn::BindingPointInfo>& outputBindings)
        : m_NetworkId(networkId), m_Runtime(runtime), m_InputBindings(inputBindings), m_OutputBindings(outputBindings)
    {}

    static TfLiteStatus AddInputLayer(DelegateData& delegateData,
                                      TfLiteContext* tfLiteContext,
                                      const TfLiteIntArray* inputs,
                                      std::vector<armnn::BindingPointInfo>& inputBindings);

    static TfLiteStatus AddOutputLayer(DelegateData& delegateData,
                                       TfLiteContext* tfLiteContext,
                                       const TfLiteIntArray* outputs,
                                       std::vector<armnn::BindingPointInfo>& outputBindings);


    /// The Network Id
    armnn::NetworkId m_NetworkId;
    /// ArmNN Runtime
    armnn::IRuntime* m_Runtime;

    // Binding information for inputs and outputs
    std::vector<armnn::BindingPointInfo> m_InputBindings;
    std::vector<armnn::BindingPointInfo> m_OutputBindings;

};

} // armnnDelegate namespace