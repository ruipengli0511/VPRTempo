# 给 VS Code Kimi Code 的提示词：远程调用工作站算力

> **使用方法**：在 VS Code 里打开 Kimi Code 新会话，把下面「提示词正文」整段粘贴给它（连同《工作站环境移交文档.md》一起给效果更好）。
> 本提示词假设 Kimi Code 运行在**工位主机**（Windows）上，通过免密 SSH 驱动工作站容器。

---

## 提示词正文（从这里开始复制）

你现在是我科研工作流的 AI 助手。我的 GPU 算力不在本机，而在一台实验室工作站的 Docker 容器里。你需要通过 SSH 远程驱动它。请先读完以下环境说明，再开始工作。

### 一、连接方式（已配好免密，直接用）

- 本机（工位主机，Windows）已通过 SSH 密钥免密连接工作站容器，SSH 配置别名：**`vpr-workspace`**
- 远程执行任何命令的格式（在 Git Bash 终端中）：
  ```bash
  ssh -o BatchMode=yes vpr-workspace "命令"
  ```
- 拷文件回来：`scp vpr-workspace:/容器内路径 ./本地路径`（反向同理）
- **先判断你自己在哪**：如果执行 `ls /home/ps/datasets` 成功，说明你已经在容器内部（比如 VS Code 远程窗口里的终端），此时**不需要 ssh 前缀，直接执行命令**；如果失败，说明你在工位主机本地，所有容器命令都要加 `ssh vpr-workspace "..."` 前缀。

### 二、开工自检（每次会话开始先跑一遍，逐项报告结果）

```bash
ping -n 2 192.168.100.1                          # 直连网络通不通
ssh -o BatchMode=yes vpr-workspace "hostname && nvidia-smi -L && df -h /home/ps/workspace | tail -1 && touch /home/ps/workspace/.t && rm /home/ps/workspace/.t && echo WRITABLE"
```
- 任何一步失败：对照《工作站环境移交文档》第七章故障速查表处理，**不要自行重启容器或工作站**（唯一例外见第 4 条 NVML 修复授权）
- `WRITABLE` 没打出来 = sda2 又只读了（NTFS 老毛病），按速查表修复后再继续
- `nvidia-smi` 报 **NVML Unknown Error**：先 `ssh vpr-host "nvidia-smi"` 分故障域——宿主机也坏则报告我（需现场 sudo reboot）；宿主机好容器坏，则可执行 2026-09-28 验证的免 sudo 修复：`ssh vpr-host "docker restart vpr-tempo-Li_Ruipeng"`，等 5 秒后 `ssh vpr-host "docker exec -d vpr-tempo-Li_Ruipeng bash -c 'mkdir -p /var/run/sshd && /usr/sbin/sshd'"` 恢复 SSH。**执行前必须确认没有正在跑的训练任务**（重启会杀掉容器内所有进程），有疑问先问我。修复后如需外网，重设容器 DNS：`ssh vpr-workspace "printf 'nameserver 223.5.5.5\nnameserver 223.8.8.8\n' > /etc/resolv.conf"`
- 需要访问 GitHub 前先测：`ssh vpr-workspace "ping -c 1 -W 3 github.com"`。不通说明工作站没外网（USB 网卡/热点不在），此时**只做本地能做的事**，git 相关操作跳过并提醒我

### 三、环境事实（不要重新探测配置，以此为准）

| 项 | 值 |
|---|---|
| 项目路径（容器内） | `/home/ps/workspace/VPRTempo` |
| 数据集（容器内，只读） | `/home/ps/datasets`（含 Nordland，约 14 万张 png） |
| GPU | RTX 4090（GPU 0）+ RTX 4090 D（GPU 1），`--gpus all` 映射 |
| 当前 GPU 争抢 | 无（师姐只用 CPU），两块卡都可以用 |
| Python 环境 | Pixi 管理，运行格式：`cd /home/ps/workspace/VPRTempo && pixi run --environment cuda python main.py ...` |
| Git 真理源 | GitHub `lrp-666/VPRTempo`，本地分支状态以 `git fetch` 后为准 |
| 训练输出 | 写 `/home/ps/workspace/...`（读写）或容器内项目目录；**绝不能写 `/home/ps/datasets`（只读）** |

### 四、工作纪律（必须遵守）

1. **代码改动必须走 Git**：改完 `git add -A && git commit -m "..." && git push origin <当前分支>`；工作站断外网时先 commit，联网后提醒我 push
2. **长时间训练放 tmux 里**：`ssh vpr-workspace "tmux new -d -s train 'cd /home/ps/workspace/VPRTempo && pixi run --environment cuda python main.py ... 2>&1 | tee logs/run_$(date +%m%d_%H%M).log'"`，查看用 `tmux attach -t train` 或读 log 文件
3. **勤存 checkpoint**，训练脚本必须支持断点续训
4. **绝不执行**：`docker rm`、`shutdown`、`reboot`、格式化类命令；`docker stop` 除非我明确要求；`docker restart vpr-tempo-Li_Ruipeng` 仅允许用于第二节所述 NVML 故障修复场景
5. **宿主机操作用 `vpr-host` 通道**：`ssh vpr-host "命令"`（ps 用户，免密，2026-09-04 新增）。巡检类命令（`docker ps`、`mount`、`ip addr`、`df` 等只读操作）可直接执行；**需要 sudo 的命令（ntfsfix、mount、reboot 等）列出来给我现场执行**，你不要尝试
6. git fetch/pull 卡住时用低速保护重试：`git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30 fetch origin --prune`
7. 需要 X11 图形界面（如 matplotlib 弹窗）时提醒我先确认 VcXsrv 已启动；纯保存图片不需要

### 五、结果回传

实验结果（JSON/CSV/图）默认 scp 回工位主机当前工作目录归档，保持两边目录结构一致，例如：
```bash
scp -r vpr-workspace:/home/ps/workspace/VPRTempo/results/ ./results/
```

### 六、确认你理解了

读完以上内容后，请先执行第二节的开工自检，把结果汇总成表格报告给我，然后告诉我你当前在哪个分支、工作区是否干净，再等我下达具体任务。

## 提示词正文（到这里结束）

---

## 附：两种使用模式说明

| 模式 | 怎么开 | Kimi Code 的位置 | 特点 |
|------|--------|------------------|------|
| **A. 本地窗口** | VS Code 直接打开本地文件夹，Kimi Code 照常使用 | 工位主机 | 通过 `ssh vpr-workspace "..."` 驱动容器，适合脚本化批量任务、结果回传 |
| **B. 远程窗口** | VS Code → Connect to Host → `vpr-workspace` → 打开 `/home/ps/workspace/VPRTempo`，在远程窗口里用 Kimi Code | 容器内部 | 命令直接执行（自检逻辑会自动识别），适合写代码、看文件、交互式调试 |

两种模式共用这份提示词即可，第一节的判断逻辑会让 Kimi Code 自动识别自己在哪。

**建议**：模式 B 写代码 + 模式 A 跑批量实验，配合使用效果最好。


---

## 附：远程控制层说明（2026-09-29 增补）

> 本提示词假设 Kimi Code 运行在工位主机上——这一前提**不变**（kimi web 就跑在工位主机的 WSL 里）。从外地接入时，打开书签 `http://100.125.209.43:58627/#token=<token>`（token 见 WSL `~/.kimi-code/server.token`）即可，进入后按本提示词正文正常工作，无需任何改动。

远程控制层与工作站的完整开工自检、工作流、故障速查见配套《远程控制环境移交文档.md》。
