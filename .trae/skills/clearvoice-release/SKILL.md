---
name: "clearvoice-release"
description: "ClearVoice 项目的功能/修复收尾发布流程：补冒烟断言并全量回归、同步文档（功能说明/fixlog/模块头）、中文提交信息、提交并推送 GitHub。在本工作区完成代码修改且用户要求测试、写文档、提交或推送时调用。"
---

# ClearVoice 收尾发布流程

ClearVoice（d:\projsrc\clearVoice，PySide6 桌面音视频工具）每轮功能开发/缺陷修复完成后，
按以下固定顺序收尾。**不要跳步，不要自创命令变体。**

## 触发时机

- 用户说「测试 / 跑冒烟 / 写文档 / 更新文档 / 提交 / 推送 / 发布」或语义等价的请求；
- 一轮代码修改（app/ 下任何模块、smoke_test.py）完成需要收尾时。

## 步骤

### 1. 补冒烟断言并全量回归

- 新功能/新模块必须在 `smoke_test.py` 末尾（`print("\nALL TESTS PASSED")` 之前）
  新增一个编号测试项，格式：`print("N) 中文名...", end=" ", flush=True)` … `print("ok")`。
- 断言风格与现有项一致：纯函数逻辑直接 assert；涉及临时文件用
  `tempfile.mkdtemp(prefix="cv_xx_")` + `try/finally: shutil.rmtree(..., ignore_errors=True)`；
  注意文件顶部已 import 的模块（os/sys/numpy），缺啥在测试项内局部 import
  （`import shutil` / `import tempfile` 等）。
- Windows 注意：文件系统大小写不敏感，重命名测试不要用仅大小写不同的名字。
- 运行：`.venv\Scripts\python.exe -m app.main` 是启动；回归用
  `.venv\Scripts\python.exe smoke_test.py`，必须看到 `ALL TESTS PASSED`。
- PowerShell Restricted 执行策略会**偶发拦截整个命令**（PSSecurityException 红字），
  原样重试即可，不要改命令。

### 2. 同步文档（三份）

- `docs/功能说明.md`：
  - 顶部 ASCII 布局图页签列表要与实际页签数一致（现为 10 个）；
  - 新功能写进对应章节（`### N. 名称`），新增页签时后续章节编号全部顺延；
  - 文末「输出文件命名约定」表同步增改行。
- `docs/fixlog.md`：
  - 在 `## 回归基线` 之前按日期追加新节（同一天可并入当天 `## YYYY-MM-DD 标题`）；
  - 条目格式：`### [日期] 类别 | 一句话标题`，类别用 新增/变更/修复/界面/健壮性/测试/环境；
  - 内容按「需求/问题 → 实现/原因 → 验证」组织；
  - 更新文末回归基线：测试总项数、日期、末尾项清单（如 `→ 13. ...`）。
- `app/main_window.py` 模块头 docstring：页签清单/左栏说明有变化时同步。

### 3. 提交与推送

- 提交信息写到 `.git/_fix_msg.txt`（中文：首行标题，空行后编号列表列出改动点与测试结果）。
- 暂存**具体文件**，禁止 `git add -A`/`git add .`：
  `git add app/xxx.py app/main_window.py smoke_test.py docs/功能说明.md docs/fixlog.md`
- 工作区里无关的 `uv.lock` 修改**永远不要提交**，保持不动。
- 提交：`git commit -F .git/_fix_msg.txt`（不要 -m）。
- 推送：`git push origin main`。
  - 间歇性 SSL/443 超时 → 重试即可成功；
  - PowerShell 会把 git push 的 stderr 进度显示成红色「错误」，**不是失败**；
    以 `git status -sb` 输出无 `[ahead N]`（显示 `## main...origin/main`）为准。
- 提交哈希形如 `c0c9365`，完成后向用户报告：提交哈希、文件统计、推送结果、测试项数。

## 环境备忘（已验证，勿重复踩坑）

- Python 解释器：`.venv\Scripts\python.exe`；远程仓库 `https://github.com/pjtufo/clearvoice`，
  分支 `main`，GCM 凭据已缓存。
- 后台耗时任务（模型下载、ffmpeg 批处理）命令超时给足 300000ms。
