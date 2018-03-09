//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <backends/ArmComputeTensorUtils.hpp>
#include <backends/CpuTensorHandle.hpp>
#include <backends/NeonTensorHandle.hpp>
#include <backends/Workload.hpp>

#include <boost/cast.hpp>

namespace armnn
{

// Base class template providing an implementation of the Constant layer common to all data types
template <armnn::DataType DataFormat>
class NeonBaseConstantWorkload : public TypedWorkload<ConstantQueueDescriptor, DataFormat>
{
public:
    NeonBaseConstantWorkload(const ConstantQueueDescriptor& descriptor, const WorkloadInfo& info)
        : TypedWorkload<ConstantQueueDescriptor, DataFormat>(descriptor, info)
        , m_RanOnce(false)
    {
    }

    virtual void Execute() const override
    {
        using namespace armcomputetensorutils;

        // The intermediate tensor held by the corresponding layer output handler can be initialised with the
        // given data on the first inference, then reused for subsequent inferences.
        // The initialisation cannot happen at workload construction time since the ACL kernel for the next layer
        // may not have been configured at the time.
        if (!m_RanOnce)
        {
            const ConstantQueueDescriptor& data = this->m_Data;

            BOOST_ASSERT(data.m_LayerOutput != nullptr);
            arm_compute::ITensor& output =
                boost::polymorphic_downcast<NeonTensorHandle*>(data.m_Outputs[0])->GetTensor();

            switch (DataFormat)
            {
                case DataType::Float32:
                {
                    CopyArmComputeITensorData(data.m_LayerOutput->GetConstTensor<float>(), output);
                    break;
                }
                case DataType::QuantisedAsymm8:
                {
                    CopyArmComputeITensorData(data.m_LayerOutput->GetConstTensor<uint8_t>(), output);
                    break;
                }
                default:
                {
                    BOOST_ASSERT_MSG(false, "Unknown data type");
                    break;
                }
            }

            m_RanOnce = true;
        }
    }

private:
    mutable bool m_RanOnce;
};

} //namespace armnn
