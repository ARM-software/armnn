//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefElementwiseWorkload.hpp"

#include "Decoders.hpp"
#include "ElementwiseFunction.hpp"
#include "Encoders.hpp"
#include "Profiling.hpp"
#include "RefWorkloadUtils.hpp"
#include "StringMapping.hpp"
#include <ResolveType.hpp>
#include <vector>

namespace armnn
{

template <typename Functor, typename ParentDescriptor, typename armnn::StringMapping::Id DebugString>
RefElementwiseWorkload<Functor, ParentDescriptor, DebugString>::RefElementwiseWorkload(
    const ParentDescriptor& desc,
    const WorkloadInfo& info)
    : RefBaseWorkload<ParentDescriptor>(desc, info)
{
}

template <typename Functor, typename ParentDescriptor, typename armnn::StringMapping::Id DebugString>
void RefElementwiseWorkload<Functor, ParentDescriptor, DebugString>::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

template <typename Functor, typename ParentDescriptor, typename armnn::StringMapping::Id DebugString>
void RefElementwiseWorkload<Functor, ParentDescriptor, DebugString>::ExecuteAsync(
        WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

template <typename Functor, typename ParentDescriptor, typename armnn::StringMapping::Id DebugString>
void RefElementwiseWorkload<Functor, ParentDescriptor, DebugString>::Execute(
        std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, StringMapping::Instance().Get(DebugString));
    const TensorInfo& inputInfo0 = GetTensorInfo(inputs[0]);
    const TensorInfo& inputInfo1 = GetTensorInfo(inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    const TensorShape& inShape0 = inputInfo0.GetShape();
    const TensorShape& inShape1 = inputInfo1.GetShape();
    const TensorShape& outShape = outputInfo.GetShape();

    std::unique_ptr<Decoder<InType>> input0 = MakeDecoder<InType>(inputInfo0, inputs[0]->Map());
    std::unique_ptr<Decoder<InType>> input1 = MakeDecoder<InType>(inputInfo1, inputs[1]->Map());
    std::unique_ptr<Encoder<OutType>> output= MakeEncoder<OutType>(outputInfo, outputs[0]->Map());

    ElementwiseBinaryFunction<Functor>(inShape0,
                                       inShape1,
                                       outShape,
                                       *input0,
                                       *input1,
                                       *output);
}

} //namespace armnn

template class armnn::RefElementwiseWorkload<std::plus<float>,
                                            armnn::AdditionQueueDescriptor,
                                            armnn::StringMapping::RefAdditionWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::plus<int32_t>,
                                            armnn::AdditionQueueDescriptor,
                                            armnn::StringMapping::RefAdditionWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::minus<float>,
                                            armnn::SubtractionQueueDescriptor,
                                            armnn::StringMapping::RefSubtractionWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::minus<int32_t>,
                                            armnn::SubtractionQueueDescriptor,
                                            armnn::StringMapping::RefSubtractionWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::multiplies<float>,
                                            armnn::MultiplicationQueueDescriptor,
                                            armnn::StringMapping::RefMultiplicationWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::multiplies<int32_t>,
                                            armnn::MultiplicationQueueDescriptor,
                                            armnn::StringMapping::RefMultiplicationWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::divides<float>,
                                            armnn::DivisionQueueDescriptor,
                                            armnn::StringMapping::RefDivisionWorkload_Execute>;

template class armnn::RefElementwiseWorkload<std::divides<int32_t>,
                                            armnn::DivisionQueueDescriptor,
                                            armnn::StringMapping::RefDivisionWorkload_Execute>;

template class armnn::RefElementwiseWorkload<armnn::maximum<float>,
                                            armnn::MaximumQueueDescriptor,
                                            armnn::StringMapping::RefMaximumWorkload_Execute>;

template class armnn::RefElementwiseWorkload<armnn::maximum<int32_t>,
                                            armnn::MaximumQueueDescriptor,
                                            armnn::StringMapping::RefMaximumWorkload_Execute>;

template class armnn::RefElementwiseWorkload<armnn::minimum<float>,
                                            armnn::MinimumQueueDescriptor,
                                            armnn::StringMapping::RefMinimumWorkload_Execute>;

template class armnn::RefElementwiseWorkload<armnn::minimum<int32_t>,
                                            armnn::MinimumQueueDescriptor,
                                            armnn::StringMapping::RefMinimumWorkload_Execute>;
