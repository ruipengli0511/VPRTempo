# VPRTempo Git 工作流 · 会话交接文档

> 生成时间：2026-08-29
> 用途：跨会话迁移上下文。新会话中直接把本文档交给 AI 即可继续工作。

---

## 一、用户背景

- 工科研究生，研究项目：VPRTempo（视觉位置识别，SNN 相关，CNN+STDP 方向）
- 三台设备：
  1. **宿舍笔记本**（Windows）
  2. **工位主机**（Windows）
  3. **工作站**（通过 SSH / WSL 访问，Ubuntu，路径 `/home/ubuntu/projects/VPRTempo`，Python 3.10.12）
- 同步方式：代码走 Git（GitHub），其余文件夹走百度网盘同步空间
- ⚠️ Git 仓库**不要**放进百度网盘同步空间，避免 `.git` 损坏

---

## 二、仓库信息

- 仓库地址：`https://github.com/lrp-666/VPRTempo` ⚠️ **2026-09-29 起已迁移至 `https://github.com/ruipengli0511/VPRTempo`**，本文档其余内容仍为 2026-08-29 历史快照
- 远程名：`origin`，fetch/push 均为上述地址（已确认）
- 默认分支：`main`（`origin/HEAD -> origin/main`）

---

## 三、工作流设计

**"基线 + 多实验分支 + 消融合并" 模式：**

1. `main` 作为基线（baseline）
2. 多个独立实验分支并行做不同方向的尝试
3. 消融阶段创建 `ablation/xxx` 分支做两两组合、全组合合并

### 分支命名规范（前缀约定）

Git 中 `/` 只是普通字符，但工具会按前缀分组显示，像文件夹一样。

| 前缀 | 含义 | 用途 |
|------|------|------|
| `try/` | 尝试 / 实验 | 独立方向的实验分支 |
| `test/` | 测试 | 跑测试用（如 noshuffle） |
| `ablation/` | 消融组合 | 后续两两合并、全组合 |
| `feat/` | 新功能 | 正式新功能开发 |

---

## 四、当前分支状态（截至 2026-07-18，工作站视角）

| 本地分支 | 提交 | 追踪的远程分支 | 说明 |
|----------|------|----------------|------|
| main | 27cb579 | origin/main | ✅ 基线，已对齐远程 |
| cnn-stdp-try | 27cb579 | origin/cnn-stdp-try | ✅ 由 try/sequence-try 改名而来（CNN+STDP 主方案） |
| test/noshuffle-test | ceab175 | origin/test/noshuffle-test | ✅ 未改动，"noshufle_已跑" |
| try/encoding-try | 67148d5 | origin/try/encoding-try | ✅ 编码实验，含 "docs: 添加论文思路说明"（papers/ideas_explain/ 下 4 个文档） |
| try/network-try | 27cb579 | origin/try/network-try | ✅ 网络结构实验，仍是基线 |

**GitHub 远程现有分支：**
```
main
cnn-stdp-try
test/noshuffle-test
try/encoding-try
try/network-try
```

### 重要历史变更（2026-07-18 完成）

1. 工作站（WSL）上通过 kimi-cli 执行了分支对齐：
   - `main`、`try/network-try`、`try/sequence-try` 均 `git reset --hard origin/...` 对齐远程
   - `test/noshuffle-test` 保持不动
2. `try/sequence-try` 改名为 `cnn-stdp-try`（本地 + 远程均已完成）：
   - 远程旧分支 `try/sequence-try` 已删除
   - 新分支 `cnn-stdp-try` 已推送并绑定追踪

### ⚠️ 已知遗留问题

1. **`cnn-stdp-try` 没有 `try/` 前缀**，不符合命名规范（有意保留则无视）。如需统一：
   ```bash
   git checkout cnn-stdp-try
   git branch -m try/cnn-stdp-try
   git push origin -d cnn-stdp-try
   git push -u origin try/cnn-stdp-try
   ```
2. **宿舍笔记本和工位主机还残留旧的 `try/sequence-try`**。下次在这两台设备上操作：
   ```bash
   git fetch --prune                # 清理已删除的远程引用
   git branch -d try/sequence-try   # 删除本地旧分支（无独有提交，安全）
   git switch cnn-stdp-try          # 需要时切换到新分支
   ```

---

## 五、常用命令速查

### 创建分支（从 main 切出）
```bash
git checkout main
git checkout -b 新分支名
```

### 首次推送（绑定 upstream）
```bash
git push -u origin 分支名
```

### 日常推送
```bash
git add .
git commit -m "说明"
git push        # 已绑定 upstream，无需指定分支
```

### 日常拉取
```bash
git switch 分支名
git pull        # 只拉取当前分支对应的远程分支，不会拉 main
```

### 跨设备同步流程
- 设备 A：工作 → `git add .` → `git commit` → `git push`
- 设备 B：`git fetch origin` → `git switch 分支名` → `git pull`

### 对齐远程（强制一致）
```bash
git fetch origin                 # 先刷新远程快照！
git checkout 分支名
git reset --hard origin/分支名
```
注意：`origin/分支名` 是**上次 fetch 时的快照**，不是 GitHub 实时状态。要 100% 对齐最新代码必须先 `git fetch origin`。

### 丢弃本地修改
```bash
git restore .    # 或 git stash
```

### 分支改名（本地 + 远程完整流程）
```bash
git checkout 旧分支名
git branch -m 新分支名            # 本地改名
git push origin -d 旧分支名       # 删除远程旧分支
git push -u origin 新分支名       # 推送新分支并绑定追踪
# 其他设备：git fetch --prune
```

---

## 六、已讲解的关键概念

### `git branch` 系列命令区别

| 命令 | 显示内容 |
|------|---------|
| `git branch` | 本地分支（真正工作的分支） |
| `git branch -r` | 远程跟踪分支快照（只读缓存，如 `origin/main`） |
| `git branch -vv` | 本地分支 + 各自追踪的远程分支及提交 |

### 远程跟踪分支（`origin/xxx`）的本质

- 是 Git 在本地保存的"远程仓库长什么样"的**缓存快照**
- 只在 `git fetch` / `git pull` 时更新，不是实时的
- `origin/HEAD -> origin/main` 表示远程默认分支是 main
- 不能直接在上面提交代码

---

## 七、操作纪律（重要）

1. **切换分支前必须干净**：`git status` 确认无未提交修改
2. **不要跨分支推送**：在 A 分支不要 `git push origin B`
3. **不要在实验分支上 `git pull origin main`**：会把 main 合并进实验分支
4. `git pull` 只拉当前分支的远程对应分支
5. `-u` 是绑定 upstream，不是设置默认拉 main
6. 网络报错 `GnuTLS recv error` 时：开代理后重试即可（已验证有效）

---

## 八、待办 / 后续计划

- [ ] 在 `try/network-try` 上进行网络结构实验
- [ ] 在 `cnn-stdp-try` 上进行 CNN+STDP 实验（原 sequence-try 方向）
- [ ] 三个分支都成功后，创建 `ablation/encoding+network` 等分支两两组合
- [ ] 最后创建 `ablation/full-combine` 全组合
- [ ] 消融合并时灵活决定 `papers/ideas_explain/` 是否带入
- [ ] 宿舍笔记本、工位主机执行旧分支清理（见第四节遗留问题 2）
- [ ] （可选）决定是否给 `cnn-stdp-try` 补 `try/` 前缀

---

## 九、给新会话 AI 的说明

- 本文档涵盖了此前两个 AI 会话的全部关键上下文
- 用户习惯：在 VSCode / kimi-cli（工作站 WSL）或本机执行 git 操作，遇到具体问题会贴终端输出来问
- 回答请用中文，命令直接给出可复制版本
- 涉及 git 变更操作（reset/push/删分支等）时，先给命令让用户确认执行，或明确请求后再操作
