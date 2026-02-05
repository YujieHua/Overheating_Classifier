# Claude Guidelines

```
╔════════════════════════════════════════════════════════════════════════════╗
║                     ⚠️  STOP - READ THESE RULES FIRST  ⚠️                   ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  PROJECT STATUS: PLANNING PHASE                                            ║
║  ─────────────────────────────────                                         ║
║  Based on Ali's recommendations from Jan 27, 2026 meeting.                 ║
║  BEFORE implementing code, must:                                           ║
║  1. Review the simplified energy approach in PLAN.md                       ║
║  2. Confirm validation data from Ali (manifolds geometry + OT results)     ║
║  3. Discuss refinements with user                                          ║
║                                                                            ║
║  DO NOT start coding until user explicitly approves.                       ║
║                                                                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  RULE 1: NEW PORT ON EVERY CODE CHANGE                                     ║
║  ─────────────────────────────────────                                     ║
║  Before showing user ANY URL, you MUST:                                    ║
║  1. Kill the old server                                                    ║
║  2. Generate a NEW random port (6000-9000)                                 ║
║  3. Start server on the NEW port                                           ║
║  4. Give user the NEW URL                                                  ║
║                                                                            ║
║  WHY: Browser caches aggressively. Same port = stale page.                 ║
║                                                                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  RULE 2: ALWAYS USE STATUS TEMPLATES                                       ║
║  ────────────────────────────────────                                      ║
║  When reporting status to user, you MUST use the exact templates below:    ║
║  • ⏳ NEEDS YOUR INPUT  - when you need a decision                         ║
║  • ✅ READY FOR REVIEW  - when done, awaiting user review                  ║
║  • 🎉 MERGED & COMPLETE - after PR merged and cleanup done                 ║
║  • 🚫 BLOCKED           - when you cannot proceed                          ║
║                                                                            ║
║  WHY: User runs parallel sessions. Templates let them scan quickly.        ║
║                                                                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  RULE 3: NEVER PUSH DIRECTLY - ALWAYS USE PR                               ║
║  ───────────────────────────────────────────                               ║
║  No matter how small the change:                                           ║
║  1. Commit to a feature branch (NEVER to main)                             ║
║  2. Push the feature branch to origin                                      ║
║  3. Create a Pull Request                                                  ║
║  4. Merge via PR (NEVER git push origin main)                              ║
║                                                                            ║
║  WHY: PRs keep a record of all changes with context and review.            ║
║  Even typo fixes get PRs. No exceptions.                                   ║
║                                                                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  ⚡⚡⚡ RULE 4: VERIFY BEFORE REPORTING - NON-NEGOTIABLE ⚡⚡⚡              ║
║  ─────────────────────────────────────────────────────────────             ║
║  BEFORE telling user "it works" or "ready for review":                     ║
║                                                                            ║
║  1. RUN IT - Start server, run simulation, execute the code                ║
║  2. SEE IT - Take screenshots, capture output, view the result             ║
║  3. TEST IT - Click buttons, submit forms, trigger the feature             ║
║  4. PROVE IT - Include evidence (screenshot, output, test result)          ║
║                                                                            ║
║  ┌──────────────────────────────────────────────────────────────┐          ║
║  │  "I made the change" is NOT enough.                         │          ║
║  │  "I verified it works" WITH PROOF is required.              │          ║
║  └──────────────────────────────────────────────────────────────┘          ║
║                                                                            ║
║  WHY: Users waste time debugging "completed" work that was never tested.   ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
```

---

## Project Info

- **Project**: Overheating Classifier (Simplified energy-based overheating indicator)
- **GitHub**: https://github.com/YujieHua/Overheating_Classifier
- **Local Folder**: `C:\Users\huayu\Local\Desktop\Overheating_Classifier\`
- **Related Project**: `C:\Users\huayu\Local\Desktop\Overheating_Predictor\` (Temperature-based model)

**Project Type**: Python backend + web interface

**Origin**: Based on Ali's recommendations from Jan 27, 2026 SRG meeting. This is a simplified alternative to the temperature-based Overheating Predictor project.

---

## Relationship to Other Projects

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TWO PARALLEL PROJECTS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Overheating_Predictor/           Overheating_Classifier/                   │
│  ─────────────────────            ───────────────────────                   │
│  • Temperature-based              • Energy-based (THIS PROJECT)             │
│  • Rosenthal × Geometry Mult.     • Joules in/out balance                   │
│  • Time-stepped simulation        • Simple accumulation tracking            │
│  • ~80-90% accuracy target        • ~60-70% accuracy (Ali's estimate)       │
│  • Academic/research focus        • Industry/practical focus                │
│  • Complex physics                • Simple physics                          │
│                                                                              │
│  Both use: Geometry Multiplier (3D Gaussian convolution)                    │
│  Both validate against: OT data from SmartFusion                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
C:\Users\huayu\Local\Desktop\Overheating_Classifier\
├── CLAUDE.md                # This file
├── PLAN.md                  # Methodology document (energy-based)
├── MEETING_NOTES_2026-01-27.md  # Origin meeting notes
├── Validation_Data/         # OT data from Ali (TO BE RECEIVED)
│   ├── manifolds_geometry.stl
│   └── OT_results/
├── base-repo/               # Git base for worktrees (READY)
└── workspaces/              # Claude creates worktrees here
    └── 2026-XX-XX-XXXX-task\
```

---

## Setup Status: READY

**GitHub repo:** https://github.com/YujieHua/Overheating_Classifier
**base-repo:** Cloned and ready
**workspaces/:** Created

**Before starting implementation:**
- [x] Create GitHub repo
- [x] Clone to base-repo
- [ ] Receive validation data from Ali
- [ ] User approves PLAN.md

---

## Key Documents

| Document | Location | Purpose |
|----------|----------|---------|
| **PLAN.md** | This folder | Energy-based methodology |
| **MEETING_NOTES_2026-01-27.md** | This folder | Meeting where Ali proposed this approach |
| **Validation_Data/** | This folder | OT data from Ali (pending) |

---

## Quick Start for Claude (After Setup Complete)

1. **Understand the task** from user
2. **Run** `/rename task-description` (for session resumption)
3. **Check current phase** - planning or implementation?
4. If implementation approved: **Run the setup commands below**
5. **Work in the worktree**, create PR when done
6. **After PR merged**: Run cleanup commands (MANDATORY)

---

## Rule A: Git Worktree Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  WHY THIS WORKS:                                                │
│  • git fetch origin = Downloads LATEST commits from GitHub      │
│  • git worktree add ... origin/main = Creates from REMOTE       │
│  • You ALWAYS get the latest code, not outdated local code      │
└─────────────────────────────────────────────────────────────────┘
```

### Starting a New Task (Claude MUST run these)

```powershell
# 1. Set task description
$TASK_DESC = "your-task-description"  # e.g., "implement-energy-calc"
$TIMESTAMP = Get-Date -Format "yyyy-MM-dd-HHmm"
$WORKSPACE_NAME = "$TIMESTAMP-$TASK_DESC"

# 2. Define terminal title function and set initial title
function Set-TerminalTitle($title) { $Host.UI.RawUI.WindowTitle = $title; [Console]::Write("`e]0;$title`a") }
Set-TerminalTitle "$TASK_DESC (starting)"

# 3. Fetch LATEST from GitHub (this gets newest code)
Set-Location "C:\Users\huayu\Local\Desktop\Overheating_Classifier\base-repo"
git fetch origin

# 4. Create worktree from origin/main (REMOTE = latest GitHub code)
git worktree add "..\workspaces\$WORKSPACE_NAME" -b "feature/$TASK_DESC" origin/main

# 5. Enter worktree
Set-Location "..\workspaces\$WORKSPACE_NAME"
```

### After PR is Merged (Claude MUST run these for cleanup)

```powershell
# 1. Stop server if running
Get-NetTCPConnection -LocalPort $PORT -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }

# 2. Return to base repo
Set-Location "C:\Users\huayu\Local\Desktop\Overheating_Classifier\base-repo"

# 3. Remove worktree
git worktree remove "..\workspaces\$WORKSPACE_NAME" --force

# 4. Delete the merged branch
git branch -d "feature/$TASK_DESC"

# 5. Prune stale entries
git worktree prune

# 6. Play completion sound
(New-Object System.Media.SoundPlayer "C:\Windows\Media\tada.wav").PlaySync()
```

---

## Rule B: Server Restart (NEW PORT EVERY TIME)

**Initial server start:**
```powershell
do {
    $PORT = Get-Random -Minimum 6000 -Maximum 9000
} while (Get-NetTCPConnection -LocalPort $PORT -ErrorAction SilentlyContinue)
Write-Host "Using port $PORT"
Set-TerminalTitle "$TASK_DESC (working, port $PORT)"
python app.py --port=$PORT
```

**After ANY code change - MUST use new port:**
```powershell
Get-NetTCPConnection -LocalPort $PORT -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
do {
    $PORT = Get-Random -Minimum 6000 -Maximum 9000
} while (Get-NetTCPConnection -LocalPort $PORT -ErrorAction SilentlyContinue)
Write-Host "Restarting on NEW port $PORT"
Set-TerminalTitle "$TASK_DESC (working, port $PORT)"
python app.py --port=$PORT
```

---

## Rule C: Audio Notifications

| Status | Sound | When |
|--------|-------|------|
| NEEDS INPUT | `Windows Notify.wav` | Need decision to proceed |
| READY FOR REVIEW | `chimes.wav` | Done, awaiting review |
| COMPLETE | `tada.wav` | PR merged, cleanup done |
| BLOCKED | `Windows Critical Stop.wav` | Error, cannot proceed |

```powershell
# NEEDS INPUT
Set-TerminalTitle "$TASK_DESC (needs input, port $PORT)"
(New-Object System.Media.SoundPlayer "C:\Windows\Media\Windows Notify.wav").PlaySync()

# READY FOR REVIEW
Set-TerminalTitle "$TASK_DESC (ready for review, port $PORT)"
(New-Object System.Media.SoundPlayer "C:\Windows\Media\chimes.wav").PlaySync()

# COMPLETE
Set-TerminalTitle "$TASK_DESC (complete)"
(New-Object System.Media.SoundPlayer "C:\Windows\Media\tada.wav").PlaySync()

# BLOCKED
Set-TerminalTitle "$TASK_DESC (blocked, port $PORT)"
(New-Object System.Media.SoundPlayer "C:\Windows\Media\Windows Critical Stop.wav").PlaySync()
```

**Terminal title in bash/MSYS (Claude Code environment):**
```bash
echo -ne "\033]0;Energy calc (working, port 7234)\007"
```

---

## Rule D: Ask Before Major Decisions

Use `AskUserQuestion` tool for:
- Architectural decisions (new modules, patterns, restructuring)
- Physics model changes (different equations, new parameters)
- When the user's goal is unclear

Don't ask for: straightforward implementations from approved plan, routine changes.

---

## Rule E: Use PowerShell for Windows-Specific Operations

**Archive operations (zip/unzip):**
```powershell
# Create ZIP
powershell.exe -Command "Compress-Archive -Path 'C:\path\to\file' -DestinationPath 'C:\path\to\archive.zip' -Force"

# Extract ZIP
powershell.exe -Command "Expand-Archive -Path 'C:\path\to\archive.zip' -DestinationPath 'C:\path\to\folder' -Force"
```

---

## Rule F: VERIFY CHANGES BEFORE REPORTING (CRITICAL)

| Change Type | Required Verification |
|-------------|----------------------|
| **Web UI changes** | Start server, open in browser, take screenshot |
| **Backend/API changes** | Run the endpoint, show the response |
| **Simulation/calculation** | Run simulation, show output values |
| **Bug fixes** | Reproduce bug first, then show it's fixed |
| **New features** | Demo the feature working end-to-end |
| **Refactoring** | Run existing tests, show they still pass |

### Rule F.1: USE CHROME EXTENSION TO RUN SIMULATIONS (NON-NEGOTIABLE)

```
┌──────────────────────────────────────────────────────────────────┐
│  When presenting a test URL to the user, do NOT just give the    │
│  URL and ask the user to run the simulation themselves.           │
│                                                                  │
│  INSTEAD: Use the Chrome browser automation extension             │
│  (mcp__claude-in-chrome__*) to:                                  │
│  1. Navigate to the test URL                                     │
│  2. Fill in any required inputs / upload test files               │
│  3. Click buttons to run the simulation end-to-end               │
│  4. Wait for results to appear                                   │
│  5. Take screenshots of the final result                         │
│  6. Present the screenshots as proof                             │
│                                                                  │
│  WHY: User should see the finished result, not do manual work    │
│  to verify something Claude was supposed to test.                │
└──────────────────────────────────────────────────────────────────┘
```

**CRITICAL REQUIREMENTS:**

1. **Always verify using Chrome extension BEFORE presenting to user**
   - Never say "here's the URL, try it out"
   - Always open the URL yourself using `mcp__claude-in-chrome__navigate`
   - Always run any simulations or analyses end-to-end
   - Always capture screenshots showing the final result

2. **When presenting URLs, they must ALREADY be open in Chrome**
   - Use `mcp__claude-in-chrome__tabs_context_mcp` to check tabs
   - Use `mcp__claude-in-chrome__navigate` to open the URL
   - Verify the page loaded successfully before presenting to user

3. **Complete simulations/analyses before reporting**
   - Don't ask user to "click Run" or "start the analysis"
   - Use `mcp__claude-in-chrome__computer` or `mcp__claude-in-chrome__form_input` to click buttons
   - Wait for results to complete
   - Take screenshots showing completed results
   - Present completed work, not work-in-progress

**Example workflow:**
```
1. Start server on new port (e.g., 7234)
2. Use mcp__claude-in-chrome__navigate to http://localhost:7234
3. If simulation has inputs, use mcp__claude-in-chrome__form_input to fill them
4. Use mcp__claude-in-chrome__computer to click "Run Simulation"
5. Wait for completion (check for results appearing)
6. Use mcp__claude-in-chrome__read_page to verify results are present
7. Take screenshot with completed results
8. THEN present to user: "✅ Simulation complete, see screenshot"
```

**What NOT to do:**
- ❌ "Here's the URL: http://localhost:7234 - try uploading a file"
- ❌ "Server is running, you can test it now"
- ❌ "Navigate to the page and click Run"

**What TO do:**
- ✅ "I've run the simulation end-to-end. See screenshot showing [specific result]"
- ✅ "Analysis complete. Results show [specific findings]. Screenshot attached."
- ✅ "Tested with sample data. Here's what happens [screenshot of completed run]"

---

## Status Templates (MANDATORY)

### NEEDS YOUR INPUT
```
╔══════════════════════════════════════════════════════════════════╗
║  ⏳ NEEDS YOUR INPUT                                              ║
╠══════════════════════════════════════════════════════════════════╣
║  QUESTION:                                                       ║
║  [What decision/input is needed]                                 ║
║                                                                  ║
║  OPTIONS:                                                        ║
║  1. [Option A] - [trade-off]                                     ║
║  2. [Option B] - [trade-off]                                     ║
╚══════════════════════════════════════════════════════════════════╝
```

### READY FOR REVIEW
```
╔══════════════════════════════════════════════════════════════════╗
║  ✅ READY FOR REVIEW                                              ║
╠══════════════════════════════════════════════════════════════════╣
║  TEST URL: http://localhost:XXXX/                                ║
╠══════════════════════════════════════════════════════════════════╣
║  WHAT CHANGED:                                                   ║
║  • [Change 1]                                                    ║
║  • [Change 2]                                                    ║
╠══════════════════════════════════════════════════════════════════╣
║  ⚡ VERIFICATION PERFORMED:                                       ║
║  • [What I ran/tested]                                           ║
║  • [Actual result observed]                                      ║
║  • [Evidence attached]                                           ║
╠══════════════════════════════════════════════════════════════════╣
║  HOW YOU CAN VERIFY:                                             ║
║  1. [Action to take]                                             ║
║  2. [What to expect]                                             ║
╠══════════════════════════════════════════════════════════════════╣
║  PR: https://github.com/YujieHua/Overheating_Classifier/pull/XX  ║
╚══════════════════════════════════════════════════════════════════╝
```

### MERGED & COMPLETE
```
╔══════════════════════════════════════════════════════════════════╗
║  🎉 MERGED & COMPLETE                                             ║
╠══════════════════════════════════════════════════════════════════╣
║  SUMMARY:                                                        ║
║  • [What was accomplished]                                       ║
╠══════════════════════════════════════════════════════════════════╣
║  CLEANUP: Worktree removed, branch deleted, server stopped       ║
╚══════════════════════════════════════════════════════════════════╝
```

### BLOCKED
```
╔══════════════════════════════════════════════════════════════════╗
║  🚫 BLOCKED                                                       ║
╠══════════════════════════════════════════════════════════════════╣
║  PROBLEM:                                                        ║
║  [What's blocking progress]                                      ║
╠══════════════════════════════════════════════════════════════════╣
║  NEED FROM YOU:                                                  ║
║  [Action required to unblock]                                    ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Git Workflow

- Always work on feature branches (never commit to main)
- **NEVER run `git push origin main`** - always push feature branch and create PR
- Merge only via PR with user approval
- Test code before creating PR
- Even single-line typo fixes require a PR
