//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnn
{

using NetworkId = int;

namespace experimental
{

struct ExecutionData;

struct WorkingMemDescriptor;

class IWorkingMemHandle
{
public:
    virtual ~IWorkingMemHandle() {};

    /// Returns the NetworkId of the Network that this IWorkingMemHandle works with.
    virtual NetworkId GetNetworkId() = 0;

    /// Allocate the backing memory required for execution. If this is not called, then allocation will be
    /// deferred to execution time.
    virtual void Allocate() = 0;

    /// Free the backing memory required for execution.
    virtual void Free() = 0;

    /// IsAllocated returns true if the backing memory is currently allocated.
    virtual bool IsAllocated() = 0;

    /// Get the WorkingMemDescriptor at an index. The WorkingMemDescriptors are stored in the same order as
    /// the Workloads in a topologically sorted graph.
    virtual WorkingMemDescriptor& GetWorkingMemDescriptorAt(unsigned int id) = 0;

    /// Get the ExecutionData at an index.
    /// The ExecutionData is paired with a BackendId to be able to call backend specific functions upon it.
    /// The ExecutionData are stored in the same order as the Workloads in a topologically sorted graph.
    virtual std::pair<BackendId, ExecutionData>& GetExecutionDataAt(unsigned int id) = 0;
};

} // end experimental namespace

} // end armnn namespace
