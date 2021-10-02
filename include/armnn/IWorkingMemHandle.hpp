//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

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

    /// Allocate the backing memory required for execution. If this is not called, then allocation will be
    /// deferred to execution time.
    virtual void Allocate() = 0;

    /// Free the backing memory required for execution.
    virtual void Free() = 0;

    /// IsAllocated returns true if the backing memory is currently allocated.
    virtual bool IsAllocated() = 0;

    /// Get the WorkingMemDescriptor for a Layer.
    virtual WorkingMemDescriptor& GetWorkingMemDescriptor(LayerGuid id) = 0;

    /// Get the WorkingMemDescriptor at an index. The WorkingMemDescriptors are stored in the same order as
    /// the Workloads in a topologically sorted graph.
    virtual WorkingMemDescriptor& GetWorkingMemDescriptorAt(unsigned int id) = 0;
};

} // end experimental namespace

} // end armnn namespace
