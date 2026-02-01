# Project Relationship: Two Parallel Overheating Models

## Overview

This document explains the relationship between two parallel projects for LPBF overheating prediction.

---

## The Two Projects

### 1. Overheating_Predictor (Temperature-Based)
**Location:** `C:\Users\huayu\Local\Desktop\Overheating_Predictor\`
**GitHub:** https://github.com/YujieHua/AM_Overheating_Predictor

- **Approach:** Time-stepped temperature simulation
- **Physics:** Superposed Rosenthal × Geometry Multiplier
- **Output:** Actual temperature (°C) per layer
- **Accuracy:** ~80-90% (target)
- **Complexity:** Medium-High
- **Focus:** Academic/research, detailed physics

### 2. Overheating_Classifier (Energy-Based)
**Location:** `C:\Users\huayu\Local\Desktop\Overheating_Classifier\`
**GitHub:** https://github.com/YujieHua/Overheating_Classifier

- **Approach:** Simple energy balance tracking
- **Physics:** Joules in - Joules out = Accumulated
- **Output:** Relative risk indicator (Joules)
- **Accuracy:** ~60-70% (Ali's estimate)
- **Complexity:** Low
- **Focus:** Industry/practical, quick screening

---

## Why Two Projects?

| Temperature Model | Energy Model |
|-------------------|--------------|
| Your original research approach | Ali's recommended simpler approach |
| More physics, more accurate | Simpler, faster to validate |
| Novel contribution (Rosenthal + G) | Quick win for industry partner |
| Longer development | Can be done during 3-4 week window |

**Decision:** Work on both in parallel, as separate projects.

---

## Shared Components

Both projects can share:
- **STL Slicer** - Same slicing logic
- **Geometry Multiplier (G)** - Same 3D Gaussian convolution
- **Validation Data** - Same OT data from Ali

---

## Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  When working on TEMPERATURE model:                                          │
│  → Open: C:\Users\huayu\Local\Desktop\Overheating_Predictor\                │
│  → GitHub: https://github.com/YujieHua/AM_Overheating_Predictor             │
│  → Reference: PLAN.md (Rosenthal × Geometry Multiplier)                      │
│                                                                              │
│  When working on ENERGY model:                                               │
│  → Open: C:\Users\huayu\Local\Desktop\Overheating_Classifier\               │
│  → GitHub: https://github.com/YujieHua/Overheating_Classifier               │
│  → Reference: PLAN.md (Joules accumulation)                                  │
│                                                                              │
│  Both use same CLAUDE.md workflow (worktrees, PRs, verification)             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Origin

- **Temperature Model:** Developed from Prof. Toyserkani's approval (Jan 21, 2026)
- **Energy Model:** Based on Ali's recommendations (Jan 27, 2026 SRG meeting)

Both models address the same problem with different approaches.
