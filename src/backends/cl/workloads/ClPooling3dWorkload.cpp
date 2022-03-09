//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClPooling3dWorkload.hpp"
#include <cl/ClLayerSupport.hpp>
#include <cl/ClTensorHandle.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
    using namespace armcomputetensorutils;

    arm_compute::Status ClPooling3dWorkloadValidate(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    const Pooling3dDescriptor& descriptor)
    {
        const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
        const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);

        arm_compute::Pooling3dLayerInfo layerInfo = BuildArmComputePooling3dLayerInfo(descriptor);

        return arm_compute::CLPooling3dLayer::validate(&aclInputInfo, &aclOutputInfo, layerInfo);
    }

    ClPooling3dWorkload::ClPooling3dWorkload( const Pooling3dQueueDescriptor& descriptor,
                                              const WorkloadInfo& info,
                                              const arm_compute::CLCompileContext& clCompileContext)
                                              : ClBaseWorkload<Pooling3dQueueDescriptor>(descriptor, info)
    {
        // Report Profiling Details
        ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClPooling3dWorkload_Construct",
                                             descriptor.m_Parameters,
                                             info,
                                             this->GetGuid());

        m_Data.ValidateInputsOutputs("ClPooling3dWorkload", 1, 1);

        arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
        arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

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
            ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClPooling3dWorkload_configure");
            // Run the layer.
            m_PoolingLayer.configure(clCompileContext, &input, &output, layerInfo);
        }
    }

    void ClPooling3dWorkload::Execute() const
    {
        ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClPooling3dWorkload_Execute", this->GetGuid());
        RunClFunction(m_PoolingLayer, CHECK_LOCATION());
    }

}


