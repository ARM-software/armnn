//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <mutex>

namespace armnn
{

using NetworkId = int;

namespace experimental
{

struct WorkingMemDescriptor;

class IWorkingMemHandle
{
public:
    virtual ~IWorkingMemHandle() {};

    /// Returns the NetworkId of the Network that this IWorkingMemHandle works with.
    virtual NetworkId GetNetworkId() = 0;

    /// Returns the InferenceId of the Inference that this IWorkingMemHandle works with.
    virtual profiling::ProfilingGuid GetInferenceId() = 0;

    /// Allocate the backing memory required for execution. If this is not called, then allocation will be
    /// deferred to execution time. The mutex must be locked.
    virtual void Allocate() = 0;

    /// Free the backing memory required for execution. The mutex must be locked.
    virtual void Free() = 0;

    /// IsAllocated returns true if the backing memory is currently allocated. The mutex must be locked.
    virtual bool IsAllocated() = 0;

    /// Get a mutex which can be used for synchronizing access to the WorkingMemHandle object.
    virtual std::mutex& GetMutex() = 0;

    /// Get the WorkingMemDescriptor for a Layer. The mutex must be locked.
    virtual WorkingMemDescriptor& GetWorkingMemDescriptor(LayerGuid id) = 0;

    /// Get the WorkingMemDescriptor at an index. The WorkingMemDescriptors are stored in the same order as
    /// the Workloads in a topologically sorted graph. The mutex must be locked.
    virtual WorkingMemDescriptor& GetWorkingMemDescriptorAt(unsigned int id) = 0;
};

} // end experimental namespace

} // end armnn namespace
