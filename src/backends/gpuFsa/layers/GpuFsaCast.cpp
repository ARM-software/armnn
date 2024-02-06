//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GpuFsaCast.hpp"
#include "gpuFsa/GpuFsaBackendId.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuCast.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadContext.h>
#include <arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h>

using namespace arm_compute::experimental::dynamic_fusion;

namespace armnn
{

static constexpr arm_compute::ConvertPolicy g_AclConvertPolicy = arm_compute::ConvertPolicy::SATURATE;

static CastAttributes CastAttributesFromTensorInfo(const TensorInfo& input)
{
    using namespace armcomputetensorutils;

    CastAttributes cast_attr;
    arm_compute::DataType dataType = GetArmComputeDataType(input.GetDataType(), false);
    cast_attr.data_type(dataType).convert_policy(g_AclConvertPolicy);
    return cast_attr;
}

arm_compute::Status GpuFsaCastValidate(const TensorInfo& input, const TensorInfo& output)
{
    using namespace armcomputetensorutils;

    // Create a new workload sketch, for validation purposes
    auto compileCtx         = arm_compute::CLKernelLibrary::get().get_compile_context();
    auto workloadContext    = GpuWorkloadContext(&compileCtx);
    GpuWorkloadSketch sketch{ &workloadContext };

    arm_compute::TensorInfo aclinputInfo = BuildArmComputeTensorInfo(input, input.GetNumDimensions());

    aclinputInfo.set_are_values_constant(input.IsConstant());

    arm_compute::ITensorInfo*  inputInfo0 = workloadContext.create_tensor_info(aclinputInfo);

    CastAttributes cast_attr = CastAttributesFromTensorInfo(output);

    arm_compute::Status aclStatus = GpuCast::validate_op(sketch, inputInfo0, cast_attr);
#ifndef NDEBUG
    const bool validated = aclStatus.error_code() == arm_compute::ErrorCode::OK;
    if (!validated)
    {
        std::cout << "GpuFsaCastValidate failed: " << aclStatus.error_description() << std::endl;
    }
#endif
    return aclStatus;
}

void GpuFsaCastCreateOp(GpuFsaPreCompiledBlob* blob,
                        const TensorInfo& input,
                        const TensorInfo& output)
{
    using namespace armcomputetensorutils;

    GpuWorkloadSketch* sketch           = blob->sketch.get();
    GpuWorkloadContext* workloadContext = blob->workloadContext.get();
    std::vector<arm_compute::ITensorInfo*> inputTensorInfos  = {};
    std::vector<arm_compute::ITensorInfo*> outputTensorInfos = {};

    arm_compute::TensorInfo aclinputInfo = BuildArmComputeTensorInfo(input, input.GetNumDimensions());

    aclinputInfo.set_are_values_constant(input.IsConstant());

    inputTensorInfos.emplace_back(workloadContext->create_tensor_info(aclinputInfo));

    CastAttributes cast_attr = CastAttributesFromTensorInfo(output);

    // Validate operator, check status and update reasonIfUnsupported
    arm_compute::Status aclStatus = GpuCast::validate_op(*sketch, inputTensorInfos[0], cast_attr);
    const bool validated = aclStatus.error_code() == arm_compute::ErrorCode::OK;
    if (!validated)
    {
        throw BackendCapabilityException("\"" + std::string(GpuFsaBackendId())
                                         + "\" backend failed during cast operator validation");
    }

    arm_compute::ITensorInfo* castOutputInfo =
            GpuCast::create_op(*sketch, inputTensorInfos[0], cast_attr);

    // Temporary fix until fusing attempt is make for GpuFsa backend and Output layer workload is created.
    outputTensorInfos.emplace_back(workloadContext->create_tensor_info());
    GpuOutput::create_op(*sketch, castOutputInfo, outputTensorInfos[0]);

    // Store the TensorInfos within the blob as unique_ptrs to be used later
    blob->inputTensorInfos  = std::make_unique<std::vector<arm_compute::ITensorInfo*>>(inputTensorInfos);
    blob->outputTensorInfos = std::make_unique<std::vector<arm_compute::ITensorInfo*>>(outputTensorInfos);
}

} // namespace armnn