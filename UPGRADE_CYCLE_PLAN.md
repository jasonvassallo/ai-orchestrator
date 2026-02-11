# Separate Major/Range-Bound Upgrade Cycle Plan

This plan is intentionally isolated from the completed Wave A/B safe upgrades.

## Goals

- Validate major-version and range-bound dependency upgrades without regressing CLI or VS Code extension workflows.
- Keep each risky upgrade lane independently reversible.

## Wave C.1: Node Typings Major Lane

- Target: `vscode-extension` `@types/node` `20.x -> 25.x`.
- Steps:
  - Bump only `@types/node`.
  - Run `npm run lint` and `npm run package`.
  - Validate extension command registration and activation in VS Code.
- Exit criteria:
  - No TypeScript/JavaScript lint issues.
  - Commands show up in Command Palette and execute.

## Wave C.2: Python Major-Risk Lane

- Target: major/runtime-sensitive upgrades (starting with `websockets 15 -> 16`).
- Steps:
  - Upgrade one major dependency at a time.
  - Run `ruff check src`, `mypy src`, `pytest -q`.
  - Run CLI smoke tests:
    - Music: `python -m src.orchestrator "<prompt>" --music --music-model musicgen-small --music-duration 1 --incognito`
    - MLX vision path: `python -m src.orchestrator "<prompt>" --model mlx-llama-vision-11b --incognito --max-tokens 32`
- Exit criteria:
  - Static checks and tests pass.
  - Both CLI smoke tests succeed.

## Wave C.3: Toolchain Lane

- Target: packaging/build tooling (`pip`, `setuptools`, `packaging`) in project virtualenvs.
- Steps:
  - Upgrade toolchain packages in `.venv` and `.music-venv`.
  - Re-run full validation matrix from Wave C.2 plus extension packaging.
- Exit criteria:
  - No install/runtime regressions.
  - Reproducible environment setup remains intact.

## Release Discipline

- Use one branch per lane and one PR per lane.
- No cross-lane dependency changes in the same PR.
- Rollback rule: revert lane branch if any exit criterion fails.
