//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/ILayerSupport.hpp>

namespace armnn
{

class LayerSupportBase : public ILayerSupport
{
public:

    bool IsLayerSupported(const LayerType& type,
                          const std::vector<TensorInfo>& infos,
                          const BaseDescriptor& descriptor,
                          const Optional<LstmInputParamsInfo>& lstmParamsInfo = EmptyOptional(),
                          const Optional<QuantizedLstmInputParamsInfo>& quantizedLstmParamsInfo = EmptyOptional(),
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsDetectionPostProcessSupported(const TensorInfo& boxEncodings,
                                         const TensorInfo& scores,
                                         const TensorInfo& anchors,
                                         const TensorInfo& detectionBoxes,
                                         const TensorInfo& detectionClasses,
                                         const TensorInfo& detectionScores,
                                         const TensorInfo& numDetections,
                                         const DetectionPostProcessDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsMemCopySupported(const TensorInfo& input,
                            const TensorInfo& output,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsMemImportSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsMergeSupported(const TensorInfo& input0,
                          const TensorInfo& input1,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsQuantizedLstmSupported(const TensorInfo& input,
                                  const TensorInfo& previousCellStateIn,
                                  const TensorInfo& previousOutputIn,
                                  const TensorInfo& cellStateOut,
                                  const TensorInfo& output,
                                  const QuantizedLstmInputParamsInfo& paramsInfo,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsShapeSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    bool IsStandInSupported(const std::vector<const TensorInfo*>& inputs,
                            const std::vector<const TensorInfo*>& outputs,
                            const StandInDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

};

} // namespace armnn
