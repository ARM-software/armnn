//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonLstmFloatWorkload.hpp"

namespace armnn
{
NeonLstmFloatWorkload::NeonLstmFloatWorkload(const LstmQueueDescriptor& descriptor,
                                             const WorkloadInfo& info)
        : FloatWorkload<LstmQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonLstmFloatWorkload", 1, 1);
}

void NeonLstmFloatWorkload::Execute() const
{
    throw armnn::Exception("No implementation of Lstm in the Neon backend!");
}

} // namespace armnn
