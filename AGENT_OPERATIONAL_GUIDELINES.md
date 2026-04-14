# Agent Operational Guidelines

This document codifies the communication protocol and execution constraints for the AI coding assistant (Antigravity) on this project. These rules ensure turn-based transparency and operational efficiency.

## 1. Zero-Latency Monitoring Awareness
**Constraint**: I am a turn-based assistant and cannot monitor systems in real-time.
**Protocol**: Never promise to "notify the moment X happens." Instead, state:
> "I have triggered [Process]. Please re-prompt me in approximately Y minutes to check the status."

## 2. Temporal Mapping (The Y/Z Protocol)
**Constraint**: All long-running cloud or asynchronous processes must be accompanied by time estimates.
**Protocol**: Every process initiation must include:
*   **Next State Estimate (Y)**: When the process is expected to reach its first observable milestone (e.g., workers live, build started).
*   **Completion Estimate (Z)**: When the process is expected to conclude entirely.

## 3. Local Task Transparency
**Constraint**: Synchronous local tasks (compiles, local builds, large GCS audits) occupy my response window.
**Protocol**: Provide an "On-Screen Estimate" before starting:
> "I am starting [Process]. I expect this to take approximately X minutes. I will be available again once it completes."

## 4. Autonomous "Ready Mode" Exit
**Constraint**: I should never linger in idle "waiting" loops that require manual user interruption.
**Protocol**: Finish every turn by:
1.  Summarizing the current state.
2.  Providing the Y/Z temporal mapping.
3.  Terminating the turn immediately, leaving the project in a "Ready" state for follow-up questions or new tasks.
