//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "NeonPooling3dWorkload.hpp"
#include "NeonWorkloadUtils.hpp"
#include <neon/NeonLayerSupport.hpp>
#include <neon/NeonTensorHandle.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

namespace armnn
{
    using namespace armcomputetensorutils;
    arm_compute::Status NeonPooling3dWorkloadValidate(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const Pooling3dDescriptor& descriptor)
    {
        const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
        const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);
        arm_compute::Pooling3dLayerInfo layerInfo = BuildArmComputePooling3dLayerInfo(descriptor);
        return arm_compute::NEPooling3dLayer::validate(&aclInputInfo, &aclOutputInfo, layerInfo);
    }

    NeonPooling3dWorkload::NeonPooling3dWorkload( const Pooling3dQueueDescriptor& descriptor,
                                                  const WorkloadInfo& info)
            : NeonBaseWorkload<Pooling3dQueueDescriptor>(descriptor, info)
    {
        // Report Profiling Details
        ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonPooling3dWorkload_Construct",
                                             descriptor.m_Parameters,
                                             info,
                                             this->GetGuid());

        m_Data.ValidateInputsOutputs("NeonPooling3dWorkload", 1, 1);

        arm_compute::ITensor& input = static_cast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
        arm_compute::ITensor& output = static_cast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

        arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
        input.info()->set_data_layout(aclDataLayout);
        output.info()->set_data_layout(aclDataLayout);

        // flag to use wider accumulators (32 bit instead of 16 for FP16) to improve accuracy
        // enable fp_mixed_precision for the the FP16 cases that
        // accumulation reaches a limit beyond which there is no more increment of the value
        bool fpMixedPrecision = false;

        arm_compute::Pooling3dLayerInfo layerInfo = BuildArmComputePooling3dLayerInfo(m_Data.m_Parameters,
                                                                                      fpMixedPrecision);
        {
            ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "NeonPooling3dWorkload_configure");

            auto layer = std::make_unique<arm_compute::NEPooling3dLayer>();
            layer->configure(&input, &output, layerInfo);
            m_PoolingLayer.reset(layer.release());
        }
    }
    void NeonPooling3dWorkload::Execute() const
    {
        ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonPooling3dWorkload_Execute", this->GetGuid());
        m_PoolingLayer->run();
    }
}
