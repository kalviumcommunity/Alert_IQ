# Team GitHub Workflow & Engineering Guidelines 📋

This document outlines the collaborative engineering standards, branching strategy, commit conventions, code review process, and issue tracking workflow for the **Alert_IQ** team.

---

## 1. 🌿 Branching Strategy

To maintain pipeline stability and avoid merge conflicts, our team strictly adheres to a trunk-based feature-branching model:

- **`main` Branch**:
  - Contains **production-ready and releasable code only**.
  - Direct pushes to `main` are restricted. All code must arrive via Pull Requests with passing checks and approved reviews.
- **Feature Branches**:
  - All new work (features, bug fixes, documentation, refactors) is developed on dedicated branches created from `main`.
  - **Branch Naming Pattern**: `[type]/[short-description]`
    - `feature/` - New features or capabilities (e.g., `feature/data-ingestion`, `feature/github-workflow-setup`)
    - `fix/` - Bug fixes (e.g., `fix/validation-logic`, `fix/null-pointer`)
    - `docs/` - Documentation updates (e.g., `docs/data-dictionary`, `docs/workflow-guide`)
    - `refactor/` - Code refactoring without changing functionality (e.g., `refactor/pipeline-parser`)
    - `chore/` - Maintenance, dependencies, or tooling updates (e.g., `chore/update-deps`)
- **Branch Lifecycle**:
  - Feature branches are short-lived.
  - Branches must be kept up-to-date with `main` to avoid drift.
  - Branches are **deleted immediately after merging** to keep the repository clean.

---

## 2. 📝 Commit Message Conventions

We follow the **Conventional Commits** specification to ensure clear git history and enable automated changelogs.

### Format
```text
[type]: [short summary in imperative mood]

[optional detailed body explaining what and why]
```

### Supported Types
| Type | Purpose | Example |
| :--- | :--- | :--- |
| `feat` | Adding a new feature | `feat: add data validation function` |
| `fix` | Fixing a bug | `fix: handle missing timestamps in CSV ingestion` |
| `docs` | Documentation changes | `docs: document team github workflow and conventions` |
| `refactor` | Code restructuring without behavior changes | `refactor: simplify alert routing rules` |
| `test` | Adding or updating tests | `test: add unit tests for schema validator` |
| `chore` | Dependency/config updates, tooling | `chore: update requirements.txt with validation library` |

### Why This Matters
- Provides complete context on why changes were introduced.
- Enables automated release notes and semantic versioning.
- Makes bisecting and debugging historical commits straightforward.

---

## 3. 🔍 Pull Request & Code Review Process

Pull Requests (PRs) protect production integrity and ensure shared codebase ownership.

### PR Requirements
1. **Clear Title & Context**: Use descriptive PR titles (e.g., `Add data validation workflow and team branching guidelines`).
2. **Issue Linking**: Link relevant tracking issues using keywords (e.g., `Closes #1`, `Fixes #3`).
3. **Summary & Testing Details**: Detail what changed, why, and how the changes were verified.
4. **Approval Gate**: Every PR requires **at least one approving review** before merging.

### Code Review Focus Areas
Reviewers evaluate submissions against four key pillars:
- **Correctness**: Does the code work as intended across edge cases?
- **Clarity & Maintainability**: Is the code clean, readable, and well-structured?
- **Data Integrity**: Does the change safeguard schemas, null checks, and encoding?
- **Test Coverage & Quality**: Are tests included and passing?
- **Commit History**: Are commit messages clean and following conventions?

---

## 4. 🎯 GitHub Issue Tracking Approach

Work is tracked transparently from inception to completion using GitHub Issues:

1. **Issue-First Development**: Every non-trivial task, feature, or bug fix starts with a GitHub issue before coding begins.
2. **Issue Structure**:
   - **Title**: Action-oriented and specific (e.g., *"Ingest customer transaction data into pipeline"*).
   - **Description**: Clarifies the business context, technical requirements, and explicit acceptance criteria (Definition of Done).
   - **Labels**: Categorize issues (e.g., `feature`, `documentation`, `data-pipeline`, `bug`).
   - **Assignee**: Assigned to the developer actively working on the task.
3. **Closing Lifecycle**:
   - Issues are linked directly in Pull Requests and automatically closed when the PR merges into `main`.
