//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GpuFsaLayerSupport.hpp"

#include <armnn/Types.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#if defined(ARMCOMPUTEGPUFSA_ENABLED)
#include "layers/GpuFsaCast.hpp"
#include "layers/GpuFsaConvolution2d.hpp"
#include "layers/GpuFsaDepthwiseConvolution2d.hpp"
#include "layers/GpuFsaElementwiseBinary.hpp"
#include "layers/GpuFsaPooling2d.hpp"
#include "layers/GpuFsaResize.hpp"
#endif

#include <vector>

namespace armnn
{

template<typename ... Args>
bool IsGpuFsaBackendSupported(Optional<std::string&> reasonIfUnsupported, Args... args)
{
    IgnoreUnused(reasonIfUnsupported, (args)...);
#if defined(ARMCOMPUTEGPUFSA_ENABLED)
    return true;
#else
    if (reasonIfUnsupported)
    {
        reasonIfUnsupported.value() = "The armnn library has been built without CL support";
    }
    return false;
#endif
}

#if defined(ARMCOMPUTEGPUFSA_ENABLED)
#define FORWARD_GPUFSA_LAYER_SUPPORT_FUNC(expr) (expr)
#else
#define FORWARD_GPUFSA_LAYER_SUPPORT_FUNC(expr) IsGpuFsaBackendSupported(reasonIfUnsupported)
#endif

#if defined(ARMCOMPUTEGPUFSA_ENABLED)
template<class FuncType, class... Args>
inline bool CheckIsLayerSupported(FuncType&& func, Optional<std::string&> reasonIfUnsupported, Args&&... args)
{
    arm_compute::Status aclStatus = func(std::forward<Args>(args)...);
    const bool supported = (aclStatus.error_code() == arm_compute::ErrorCode::OK);
    if (!supported && reasonIfUnsupported)
    {
        reasonIfUnsupported.value() = aclStatus.error_description();
    }
    return supported;
}

#define FORWARD_LAYER_VALIDATE_FUNC(func, reasonIfUnsupported, ...) \
    return CheckIsLayerSupported(func, reasonIfUnsupported, __VA_ARGS__);
#else
#define FORWARD_LAYER_VALIDATE_FUNC(func, reasonIfUnsupported, ...) \
    return IsGpuFsaBackendSupported(reasonIfUnsupported, __VA_ARGS__);
#endif

bool GpuFsaLayerSupport::IsLayerSupported(const LayerType& type,
                                          const std::vector<TensorInfo>& infos,
                                          const BaseDescriptor& descriptor,
                                          const Optional<LstmInputParamsInfo>& lstmParamsInfo,
                                          const Optional<QuantizedLstmInputParamsInfo>& quantizedLstmInputParamsInfo,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(lstmParamsInfo);
    IgnoreUnused(quantizedLstmInputParamsInfo);

    switch (type)
    {
        case LayerType::Cast:
        {
            if (infos.size() != 2)
            {
                throw InvalidArgumentException("Invalid number of cast TensorInfos. "
                                               "TensorInfos should be of format: {input, output}.");
            }
            FORWARD_LAYER_VALIDATE_FUNC(GpuFsaCastValidate,
                                        reasonIfUnsupported,
                                        infos[0],
                                        infos[1]);
        }
        case LayerType::Convolution2d:
        {
            if (infos.size() != 4)
            {
                throw InvalidArgumentException("Invalid number of Convolution2d TensorInfos. "
                                               "TensorInfos should be of format: {input, output, weights, biases}.");
            }

            auto desc = PolymorphicDowncast<const Convolution2dDescriptor*>(&descriptor);
            if (infos[3] == TensorInfo())
            {
                FORWARD_LAYER_VALIDATE_FUNC(GpuFsaConvolution2dValidate,
                                            reasonIfUnsupported,
                                            infos[0],
                                            *desc,
                                            infos[2],
                                            EmptyOptional());
            }
            else
            {
                FORWARD_LAYER_VALIDATE_FUNC(GpuFsaConvolution2dValidate,
                                            reasonIfUnsupported,
                                            infos[0],
                                            *desc,
                                            infos[2],
                                            infos[3]);
            }
        }
        case LayerType::DepthwiseConvolution2d:
        {
            if (infos.size() != 4)
            {
                throw InvalidArgumentException("Invalid number of DepthwiseConvolution2dDescriptor TensorInfos. "
                                               "TensorInfos should be of format: {input, output, weights, biases}.");
            }

            auto desc = PolymorphicDowncast<const DepthwiseConvolution2dDescriptor*>(&descriptor);
            if (infos[3] == TensorInfo())
            {
                FORWARD_LAYER_VALIDATE_FUNC(GpuFsaDepthwiseConvolution2dValidate,
                                            reasonIfUnsupported,
                                            infos[0],
                                            *desc,
                                            infos[2],
                                            EmptyOptional());
            }
            else
            {
                FORWARD_LAYER_VALIDATE_FUNC(GpuFsaDepthwiseConvolution2dValidate,
                                            reasonIfUnsupported,
                                            infos[0],
                                            *desc,
                                            infos[2],
                                            infos[3]);
            }
        }
        case LayerType::ElementwiseBinary:
        {
            if (infos.size() != 3)
            {
                throw InvalidArgumentException("Invalid number of ElementwiseBinary TensorInfos. "
                                               "TensorInfos should be of format: {input0, input1, output}.");
            }

            auto desc = PolymorphicDowncast<const ElementwiseBinaryDescriptor*>(&descriptor);
            FORWARD_LAYER_VALIDATE_FUNC(GpuFsaElementwiseBinaryValidate,
                                        reasonIfUnsupported,
                                        infos[0],
                                        infos[1],
                                        *desc);
        }
        case LayerType::Pooling2d:
        {
            if (infos.size() != 2)
            {
                throw InvalidArgumentException("Invalid number of Pooling2d TensorInfos. "
                                               "TensorInfos should be of format: {input, output}.");
            }

            auto desc = PolymorphicDowncast<const Pooling2dDescriptor*>(&descriptor);

            FORWARD_LAYER_VALIDATE_FUNC(GpuFsaPooling2dValidate,
                                        reasonIfUnsupported,
                                        infos[0],
                                        *desc);
        }
        case LayerType::Resize:
        {
            if (infos.size() != 2)
            {
                throw InvalidArgumentException("Invalid number of Resize TensorInfos. "
                                               "TensorInfos should be of format: {input, output}.");
            }

            auto desc = PolymorphicDowncast<const ResizeDescriptor*>(&descriptor);

            FORWARD_LAYER_VALIDATE_FUNC(GpuFsaResizeValidate,
                                        reasonIfUnsupported,
                                        infos[0],
                                        *desc);
        }
        case LayerType::Constant:
        case LayerType::Input:
        case LayerType::Output:
            return IsGpuFsaBackendSupported(reasonIfUnsupported, infos[0]);
        default:
            // Layers not supported in the GpuFsa backend.
            return false;
    }
}

} // namespace armnn