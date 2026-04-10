# Observation Tensor Specification v12.0
**Project:** Backgammon Engine (OpenSpiel)
**Status:** Finalized Blueprint
**Dimensions:** 41 planes × 1 × 24 points

## Overview
This document defines the 41-plane observation tensor used by the neural network to perceive the backgammon board. It utilizes a "Surgical Gating" logic to pivot between tactical middle-game play and pure racing logic.

## The 41-Plane Stack

| Index | Feature | Range | Goal | State (No Contact) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Raw Occupancy (Self) | Scalar (0-1) | Total checkers / 15 | **ACTIVE** |
| 2 | Raw Occupancy (Opp) | Scalar (0-1) | Opponent checkers / 15 | **ACTIVE** |
| 3 | Self Blot | Binary (0/1) | Exactly 1 checker | **ACTIVE** |
| 4 | Self Made Point | Binary (0/1) | Exactly 2 checkers | **ACTIVE** |
| 5 | Self 3-Stack | Binary (0/1) | Exactly 3 checkers | **ACTIVE** |
| 6 | Self 4-Stack | Binary (0/1) | Exactly 4 checkers | **ACTIVE** |
| 7 | Self 5-Stack | Binary (0/1) | Exactly 5 checkers | **ACTIVE** |
| 8 | Heavy Stack (Self) | Binary (0/1) | 6+ checkers | **ACTIVE** |
| 9 | Opponent Blot | Binary (0/1) | Exactly 1 checker | **ACTIVE** |
| 10 | Opponent Point | Binary (0/1) | Exactly 2 checkers | **ACTIVE** |
| 11 | Opp 3-Stack | Binary (0/1) | Exactly 3 checkers | **ACTIVE** |
| 12 | Opp 4-Stack | Binary (0/1) | Exactly 4 checkers | **ACTIVE** |
| 13 | Opp 5-Stack | Binary (0/1) | Exactly 5 checkers | **ACTIVE** |
| 14 | Heavy Stack (Opp) | Binary (0/1) | 6+ checkers | **ACTIVE** |
| 15 | Pip Count (Self) | Scalar (0-1) | Total Pips / 375 | **ACTIVE** |
| 16 | Pip Count (Opp) | Scalar (0-1) | Total Pips / 375 | **ACTIVE** |
| 17 | Race Lead | Scalar (-1 to 1) | (Self - Opp) / 375 | **ACTIVE** |
| 18 | Off-Board (Self) | Scalar (0-1) | Borne off / 15 | **ACTIVE** |
| 19 | Off-Board (Opp) | Scalar (0-1) | Borne off / 15 | **ACTIVE** |
| 20 | Moves Remaining | Scalar (0-1) | $N/4$ sub-moves left | **ACTIVE** |
| 21 | **Contact Flag** | Binary (0/1) | 1.0 = Contact; 0.0 = Race | **ACTIVE (0.0)** |
| 22 | Deep Anchors (S) | Binary (0/1) | Points 22, 23, 24 | **ZEROED** |
| 23 | Deep Anchors (O) | Binary (0/1) | Points 1, 2, 3 | **ZEROED** |
| 24 | Adv. Anchors (S) | Binary (0/1) | Points 18, 19, 20, 21 | **ZEROED** |
| 25 | Adv. Anchors (O) | Binary (0/1) | Points 4, 5, 6, 7 | **ZEROED** |
| 26 | Solid 2-Prime (S) | Binary (0/1) | Wall length $\ge 2$ | **ZEROED** |
| 27 | Solid 3-Prime (S) | Binary (0/1) | Wall length $\ge 3$ | **ZEROED** |
| 28 | Solid 4-Prime (S) | Binary (0/1) | Wall length $\ge 4$ | **ZEROED** |
| 29 | Solid 5-Prime (S) | Binary (0/1) | Wall length $\ge 5$ | **ZEROED** |
| 30 | Solid 6-Prime (S) | Binary (0/1) | Wall length $= 6$ | **ZEROED** |
| 31 | Solid 2-Prime (O) | Binary (0/1) | Opponent wall $\ge 2$ | **ZEROED** |
| 32 | Solid 3-Prime (O) | Binary (0/1) | Opponent wall $\ge 3$ | **ZEROED** |
| 33 | Solid 4-Prime (O) | Binary (0/1) | Opponent wall $\ge 4$ | **ZEROED** |
| 34 | Solid 5-Prime (O) | Binary (0/1) | Opponent wall $\ge 5$ | **ZEROED** |
| 35 | Solid 6-Prime (O) | Binary (0/1) | Opponent wall $= 6$ | **ZEROED** |
| 36 | Blockade Density (S) | Scalar (0-1) | Forward Pressure Gauge | **ZEROED** |
| 37 | Blockade Density (O) | Scalar (0-1) | Backward Containment | **ZEROED** |
| 38 | Bar Count (Self) | Scalar (0-1) | Count / 15 | **ZEROED** |
| 39 | Bar Count (Opp) | Scalar (0-1) | Count / 15 | **ZEROED** |
| 40 | Board Strength (S) | Scalar (0-1) | Home points made / 6 | **ZEROED** |
| 41 | Board Strength (O) | Scalar (0-1) | Home points made / 6 | **ZEROED** |

## Implementation Logic
The C++ `GetObservation` function must check the `HasContact()` state of the board. If false, indices 22 through 41 must be zeroed out to ensure the racing engine ignores tactical containment bias.
