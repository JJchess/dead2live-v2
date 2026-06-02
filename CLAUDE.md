# dead2live

图/文生可操控 2D 数字人。详见 [README.md](README.md)。

## Git 工作流

本仓库采用安全的自动 Git 工作流约定（由 `/setup-auto-git` 配置）：

- **仅在用户明确要求时**才执行 `commit` 或 `push`。
- 若当前处于默认分支（`main` / `master`），提交前**先创建新分支**。
- 优先**新建 commit**，而不是 `git commit --amend`。
- 执行破坏性操作（`git reset --hard`、`git push --force`、`git checkout -- .`、`git clean -fd`）前，
  必须先确认是否有更安全的替代方案，并向用户说明影响。
- **绝不**使用 `--no-verify` 跳过 git 钩子，除非用户明确要求。
- 提交信息使用清晰的**祈使句**；保持**原子提交**（一次提交只做一件事）。
