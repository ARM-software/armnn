//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonLstmFloat32Workload.hpp"

namespace armnn
{
NeonLstmFloat32Workload::NeonLstmFloat32Workload(const LstmQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info)
        : FloatWorkload<LstmQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonLstmFloat32Workload", 1, 1);
}

void NeonLstmFloat32Workload::Execute() const
{
    throw armnn::Exception("No implementation of Lstm in the Neon backend!");
}

} // namespace armnn
