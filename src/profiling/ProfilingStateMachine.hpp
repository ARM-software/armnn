//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <atomic>

#include <armnn/utility/IgnoreUnused.hpp>

namespace armnn
{

namespace  profiling
{

enum class ProfilingState
{
    Uninitialised,
    NotConnected,
    WaitingForAck,
    Active
};

class ProfilingStateMachine
{
public:
    ProfilingStateMachine() : m_State(ProfilingState::Uninitialised) {}
    ProfilingStateMachine(ProfilingState state) : m_State(state) {}

    ProfilingState GetCurrentState() const;
    void TransitionToState(ProfilingState newState);
    void Reset();

    bool IsOneOfStates(ProfilingState state1)
    {
        IgnoreUnused(state1);
        return false;
    }

    template<typename T, typename... Args >
    bool IsOneOfStates(T state1, T state2, Args... args)
    {
        if (state1 == state2)
        {
            return true;
        }
        else
        {
            return IsOneOfStates(state1, args...);
        }
    }

private:
    std::atomic<ProfilingState> m_State;
};

constexpr char const* GetProfilingStateName(ProfilingState state)
{
    switch (state)
    {
        case ProfilingState::Uninitialised: return "Uninitialised";
        case ProfilingState::NotConnected:  return "NotConnected";
        case ProfilingState::WaitingForAck: return "WaitingForAck";
        case ProfilingState::Active:        return "Active";
        default:                            return "Unknown";
    }
}

} // namespace profiling

} // namespace armnn

