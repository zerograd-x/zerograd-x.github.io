# Policy Gradient · 六种权重，一条主线

保留原网页的米白／深绿色配色、方法切换与轨迹实验。六种方法共用一个 PyTorch 训练框架，每个方法面板仅展示Actor 权重与 Critic Loss 的条件分支，并列出网络、两个 Loss、stopgrad 与流程图。

## 文件

- `index.html`：可离线打开，内嵌完整代码，下载按钮无需联网。
- `policy_gradient_demo.py`：共享 Actor / ValueCritic / QCritic、环境、rollout、returns、训练框架与compute_losses 中的六种条件分支。
- `test_policy_gradient.py`：梯度、终止、回报、训练测试。
- `build_page.py` + `template.html`：从同一份 Python 源码生成页面展示和下载内容，避免复制后不一致。

## 运行与验证

建议使用支持 PyTorch 的 Python 3.10+ 环境。

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python policy_gradient_demo.py --method 1 --episodes 300
python policy_gradient_demo.py --method 6 --episodes 300 --gamma 0.9 --seed 7
python -m unittest -v
python build_page.py
python -m http.server 8000
```

`--method` 可选 1 至 6。浏览器打开 `http://localhost:8000`；也可直接打开 `index.html`。训练使用三步有限时域玩具环境，动作 1 的期望奖励较高；不代表在复杂环境上的收敛保证。

## 数学约定

目标是起点折扣回报 `J = E[sum_t gamma^t r_t]`，状态包含时间。Actor loss 对一条轨迹的时间步求和；多条轨迹按 episode 数平均。方法 1 的权重为 `G_0`，不再乘 `gamma^t`；方法 2–6 的权重有外部 `gamma^t`。

| 方法 | 权重 w_t | Critic target |
|---|---|---|
| 1 Total Return REINFORCE | G_0 | 无 |
| 2 Reward-to-Go | gamma^t G_t | 无 |
| 3 MC baseline | gamma^t (G_t − V) | V 回归 MC G_t |
| 4 Q-based | gamma^t Q | Q 回归 MC G_t |
| 5 Advantage Q−V | gamma^t (Q − V) | Q、V 各自回归 MC G_t |
| 6 One-step TD | gamma^t (y_t − V) | y_t = r_t + gamma (1−terminated) V(next) |

Actor loss 为 `−sum_t stopgrad(w_t) log pi(a_t|s_t)`。Critic loss 为 `0.5 * mean((prediction − stopgrad(target))^2)`，方法 5 将 Q/V 两项相加。

每轮先用固定参数采样完整轨迹，使用更新前 Critic 计算 Actor 权重；Actor 和独立 Critic 各更新一次，然后丢弃轨迹。方法 3 的 baseline 不在当前轨迹上先拟合再用于当前 Actor。近似 Q 与 TD 可能引入偏差。

仅真正终止屏蔽 bootstrap。本例三步终点就是任务终点；rollout 对外部截断报错，因为方法 1–5 的纯 MC 目标需要完整后缀。迁移到持续任务时需显式选择截断 bootstrap，并承认 MC 目标不再是纯采样回报。Q Critic 评估当前策略，不是 DQN 的 max 最优控制更新。

## 部署到现有 GitHub Pages

仓库：`https://github.com/zerograd-x/zerograd-x.github.io`。

将本目录文件放到仓库的 `PolicyGradient/`，保留站点根目录和 `FlashAttention/` 原样。提交到现有 Pages 发布分支（当前源代码分支为 `main`）。沿用仓库已有 Pages 配置；若未开启，在 Settings → Pages 选择 Deploy from a branch → main → /(root)。无需 Node、前端构建或外部 CDN。

```sh
python PolicyGradient/build_page.py
python -m unittest discover -s PolicyGradient -v
git add PolicyGradient
git commit -m "Clarify six policy gradient methods with shared runnable code"
git push origin main
```

等待 GitHub Pages 部署成功后访问 `https://zerograd-x.github.io/PolicyGradient/`，确认方法 6 中出现“差异代码 · 当前 args.method = 6”。如果使用 Actions 自定义发布，沿用原工作流并确保其包含 `PolicyGradient/index.html`。
