# GitHub Copilot Instructions

## Architecture Overview

GELOS is a Python project for Geospatial Exploration of Latent Observation Space. It follows a cookiecutter data science layout with a `gelos` package containing dataset/datamodule wiring, embedding generation/extraction/transformation utilities, t-SNE computation, and plotting helpers. Model definitions live under `models/`, documentation is built with MkDocs in `docs/`, and tests live in `tests/`. Configuration and paths are centralized in `gelos/config.py`, with CLIs for embedding workflows.

## AI-Assisted Development Workflow

**Important:** Always check for `.agent-module/` directory commands before starting work. These commands provide structured workflows for:
- Feature planning and implementation
- GitHub issue and PR management
- Commit conventions and branch naming
- Testing and validation procedures

If `.agent-module/` directory exists, follow those commands for consistent workflow.

When the user explicitly invokes any of these slash-commands (for example `/plan-feature` or `/execute`), Copilot must:

- Locate the corresponding command file under `.agent-module/commands`
- Read and treat its contents as the authoritative workflow for that action
- Follow the command’s process and steps explicitly and completely to fulfill the instruction, except where doing so would conflict with higher-priority safety or system rules

When the user informally mentions a command name (for example "prime", "plan feature", "execute the plan", "run RCA"), Copilot should:

- Infer whether the user likely intends to run the corresponding .agent-module command
- If the intent is clear and appropriate in context, locate and follow that command as above
- Optionally ask for quick confirmation if intent is ambiguous (e.g., "Do you want me to run `/prime` now?")

## Assistant Behavior & Collaboration Guidelines

- Default to concise explanations; for non-trivial or multi-file work, briefly outline a numbered plan before executing changes.
- When requirements are ambiguous, ask at most 1–2 targeted clarification questions and clearly state any assumptions before proceeding.
- For substantial features or refactors, consult project docs (for example `docs/`, `.agents/`, and `.github/` files) before deciding on an approach.
- After code changes, recommend or run appropriate tests (unit, integration, or smoke) when feasible, and state exactly what was run and the outcome.
- Avoid schema, migration, or production-impacting changes (databases, secrets, billing, auth) unless the user explicitly requests them.
- Prefer small, focused changes aligned with the current request; avoid unrelated refactors unless the user approves.
- Preserve existing style and patterns in this repo unless there is an explicit decision to migrate to a new pattern.

## Validation Workflow (Docker First When Available)

If a Dockerfile or Docker Compose test service exists, prefer running lint and tests in the containerized workflow. If local tooling is missing, fall back to Docker and record the results. Capture warnings about missing env vars and ensure required `GELOS_*` variables are set when invoking Compose.

### Core PIV Loop Commands

- `/prime` - Prime the core planning/implementation loop
- `/plan-feature` - Create comprehensive implementation plans (no code)
- `/execute` - Implement from an approved plan

### Product Requirements & Documentation Commands

- `/create-prd` - Create or update a Product Requirements Document for a feature

### Git & Issue Workflow Commands

- `/commit` - Create conventional commits following project rules
- `/github-issue-create` - Create GitHub issues from plans
- `/github-feature-branch` - Create properly named feature branches
- `/github-pr-create` - Create PRs with proper linking and description

### Bug Fix Workflow Commands

- `/rca` - Perform root cause analysis for a bug
- `/implement-fix` - Implement and validate a targeted bug fix

### Validation & Review Commands

- `/code-review` - Perform structured code review
- `/code-review-fix` - Apply fixes based on review findings
- `/execution-report` - Summarize what was executed and its effects
- `/system-review` - Review system-level risks, architecture, and impacts
- `/validate` - Run validation and testing workflows

## Git Workflow for Issue Development

### ⚠️ CRITICAL: Always Work from Feature Branches

**NEVER work directly on `main` branch. Always create a feature branch for each issue.**

### Starting Work on a New Issue

Follow these steps **in order**:

**Step 1: Fetch Latest Changes**
```bash
git fetch origin
```

**Step 2: Switch to Main and Pull Latest**
```bash
git checkout main
git pull origin main
```

**Step 3: Create Feature Branch**
Create a branch that references the issue number:
```bash
git checkout -b fix/issue-XX-description
# or
git checkout -b feature/issue-XX-description
```

**Branch Naming Convention:**
- **Bug fixes**: `fix/issue-XX-short-description`
- **Features**: `feature/issue-XX-short-description`
- **Refactoring**: `refactor/issue-XX-short-description`

**Examples:**
- `fix/issue-28-populate-parameter-format`
- `feature/issue-45-supabase-subscriptions-schema`
- `refactor/issue-24-zod-peer-dependency`

**Step 4: Start Work**
Only after the feature branch is created, start making changes.

**Step 5: Code Quality Checks**
Run all quality checks before completion:
```bash
# First auto-format and commit any formatting changes
npm run format
git add .
git commit -m "style: format code"  # Only if there were formatting changes

# Then run final verification checks
npm run format:check && npm run lint && npm run build && npm test
```

**Step 6: Push & Create PR**
```bash
git push origin feature/issue-XX-description
```
- Create Pull Request on GitHub
- Reference issue number in PR description
- Request code review

**Step 7: Post-Merge Cleanup**
```bash
# Switch back to main
git checkout main
git pull origin main

# Delete branches
git branch -d feature/issue-XX-description
git push origin --delete feature/issue-XX-description
```

### Complete Workflow Example

```bash
# 1. Fetch latest
git fetch origin

# 2. Switch to main and pull
git checkout main
git pull origin main

# 3. Create feature branch (example for issue #45)
git checkout -b feature/issue-45-supabase-subscriptions-schema

# 4. Now start making changes
# ... make code changes ...

# 5. Run code quality checks before completion
npm run format                    # Auto-format code
git add .                         # Stage formatting changes
git commit -m "style: format code" # Commit formatting (only if changes made)
npm run format:check && npm run lint && npm run build && npm test

# 6. Push and create PR
git push origin feature/issue-45-supabase-subscriptions-schema
```

### Important Rules

**✅ DO:**
- Always fetch and pull main before creating branch
- Always create a feature branch before making changes
- Reference issue number in branch name
- Use descriptive branch names
- Work only on feature branches

**❌ DON'T:**
- Work directly on `main` branch
- Create branch without pulling latest main first
- Skip fetching/pulling before creating branch
- Work without an associated issue

### When No Issue Exists

**If there's no issue to work on:**
1. Create an issue first describing the work
2. Then follow the workflow above

**Never start coding without:**
- An issue describing the problem/feature
- A feature branch created from latest main

### Summary

**Complete Issue Workflow:**
1. ✅ Fetch origin
2. ✅ Checkout main & pull latest
3. ✅ Create feature branch (with issue reference)
4. ✅ Start coding & make changes
5. ✅ Run quality checks (format, lint, build, tests)
6. ✅ Push branch & create PR
7. ✅ Post-merge cleanup

**Remember:** No work without an issue, no work on main!
