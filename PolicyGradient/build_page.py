"""Build offline index.html; displayed code comes from the runnable source."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'policy_gradient_demo.py').read_text()
tree = ast.parse(source)
method_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_losses')
methods = [method_node]
branch_code = ast.get_source_segment(source, method_node)
import textwrap
setup_code, rest = branch_code.split('    # Actor：', 1)
actor_code, critic_code = ('    # Actor：' + rest).split('    # Critic：', 1)
actor_code = textwrap.dedent(actor_code).strip()
critic_code = textwrap.dedent('    # Critic：' + critic_code).strip()
critic_code = critic_code.rsplit('\nreturn weights, critic_loss', 1)[0]
snippets = [actor_code] * 6
update_code = ast.get_source_segment(source, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'update_networks'))
lines = source.splitlines(keepends=True)
excluded = {i for n in methods for i in range(n.lineno - 1, n.end_lineno)}
common = ''.join(line for i, line in enumerate(lines) if i not in excluded)
common = re.sub(r'\n{4,}', '\n\n\n', common)
page = (ROOT / 'template.html').read_text()
page = re.sub(r'const pythonCode=.*?;\nconst data=', lambda _: 'const pythonCode=' + json.dumps(source) + ';\nconst commonCode=' + json.dumps(common) + ';\nconst methodCode=' + json.dumps(snippets) + ';\nconst criticCode=' + json.dumps(critic_code) + ';\nconst setupCode=' + json.dumps(setup_code) + ';\nconst updateCode=' + json.dumps(update_code) + ';\nconst data=', page, flags=re.S)
page = page.replace("$('codeblock').textContent=pythonCode;", "$('codeblock').textContent=commonCode;")
page = page.replace('一份代码，切换六种权重。', '公用代码写一次，六种方法只看差异。')
page = page.replace('<pre id="codeblock"></pre>', '<p>上方用同一组 if/elif 对照六种 Actor 权重，用另一组 if 对照 Critic Loss；这里集中展示 Actor / ValueCritic / QCritic、环境、rollout、returns、TD target 和训练框架。下载的 .py 已合并全部函数，可以直接运行。</p><details><summary>展开共享代码（网络 → 采样 → 目标 → 更新）</summary><pre id="codeblock"></pre></details>')
page = page.replace('Actor：4 维输入 → Linear(4,64) → ReLU → Linear(64,动作数)。', '本例 Actor：3 维时间 one-hot → Linear(3,32) → Tanh → Linear(32,2) → Categorical。')
page = page.replace('图片原文件未包含在引用预览中，按请求中列出的六项组织。', '方法 3–5 使用 MC Critic，方法 6 使用一步 TD；页面代码由同一份可运行源文件生成。')
page = page.replace('完全离线 · 无依赖', '网页离线可用 · 训练需 PyTorch')
page = page.replace("$('methodnum').textContent=active+1;calculate()", "$('methodnum').textContent=active+1;renderDetails();calculate()")
extra = r'''
const networks=['无','无','ValueCritic：s → Vφ(s)','QCritic：s → Qω(s,·) → gather(a)','QCritic + ValueCritic（独立参数）','ValueCritic：s → Vφ(s)'];
const weightPaths=['完整 rewards → G₀ → wₜ = G₀','完整 rewards → Gₜ → wₜ = γᵗGₜ','完整 rewards → Gₜ；Vφ(sₜ) → wₜ = γᵗ(Gₜ − Vφ)','Qω(sₜ,aₜ) → wₜ = γᵗQω','Qω(sₜ,aₜ)、Vφ(sₜ) → wₜ = γᵗ(Qω − Vφ)','rₜ、sₜ₊₁ → yₜ = rₜ + γ(1−dₜ)Vφ(sₜ₊₁) → wₜ = γᵗ(yₜ − Vφ(sₜ))'];
const criticPaths=['没有 Critic 更新','没有 Critic 更新','Gₜ → stopgrad target；Vφ(sₜ) → LV → 更新 φ','Gₜ → stopgrad target；Qω(sₜ,aₜ) → LQ → 更新 ω','Gₜ → stopgrad target；Vφ / Qω → LV + LQ → 更新 φ、ω','yₜ → stopgrad target；Vφ(sₜ) → LV → 更新 φ'];
const criticLosses=['无','无','LV = ½ meanₜ[(Vφ(sₜ) − sg(Gₜ))²]','LQ = ½ meanₜ[(Qω(sₜ,aₜ) − sg(Gₜ))²]','LC = LQ + LV；二者各自以 sg(Gₜ) 为 MC target','LV = ½ meanₜ[(Vφ(sₜ) − sg(yₜ))²]'];
function renderDetails(){
 const block=document.createElement('section');block.className='method-detail';
 block.innerHTML=`<h3>本方法的网络与 Loss</h3><div class="table-wrap"><table><tbody><tr><th>Actor 网络</th><td>Actor：s → 32 hidden units → 2 logits → Categorical</td></tr><tr><th>Critic 网络</th><td>${networks[active]}</td></tr><tr><th>Actor Loss</th><td>LActor = −∑ₜ sg(wₜ) log πθ(aₜ|sₜ)</td></tr><tr><th>Critic Loss</th><td>${criticLosses[active]}</td></tr><tr><th>停止梯度</th><td>sg = stopgrad：整个 wₜ detach；所有 Critic target detach。只对预测支路求导。</td></tr></tbody></table></div><h3>一张图：采样、权重、两条更新支路</h3><div class="training-flow" role="img" aria-label="方法 ${active+1} 的采样与训练流程"><div class="flow-node">State sₜ → Actor θ → πθ → sample aₜ → environment → rₜ、sₜ₊₁、dₜ<br><small>sₜ₊₁ 回到下一步 State；冻结参数直到采样完成</small></div><div class="flow-arrow">↓ 保存一条新轨迹（on-policy）</div><div class="flow-node">${weightPaths[active]}</div><div class="flow-arrow">↓ stopgrad(wₜ)</div><div class="flow-node">wₜ × log πθ(aₜ|sₜ) → LActor → backward → 更新 Actor θ<br><small>log π 由 Actor 在保存的 sₜ、aₜ 上重新计算，提供可微支路</small></div><div class="flow-arrow">↓ Critic 独立支路，使用更新前预测与 target</div><div class="flow-node">${criticPaths[active]}</div><div class="flow-arrow">↓ 丢弃旧轨迹 → 用更新后的策略重新采样 ↺</div></div><h3>差异代码 · 当前 args.method = ${active+1}</h3><p class="muted">按编号对照：先选 Actor 权重，再选 Critic Loss。下面两段共同组成 compute_losses；方法 5 会同时进入 V 和 Q 两个 Critic 分支。</p><details><summary>公用输入：returns、discounts、v、q 从哪里来？</summary><pre class="setup-code"></pre></details><h3>① Actor：if / elif 选择 weights</h3><pre class="difference-code"></pre><p class="note">方法 6 的 target = r + γV(next)，weights = γᵗ(target − v)。此处一条轨迹完整且连续，最后一步真正终止，因此可以将 v 向前移一位并在末尾补零。外部截断或多条轨迹拼接不能直接照搬此写法。</p><h3>② Critic：按方法选择 critic_loss</h3><pre class="critic-code"></pre><p>方法 3、4、5 用实际 MC returns 作目标；方法 6 用停止梯度的 TD target。方法 5 的 critic_loss 是 V Loss + Q Loss。方法 1、2 保持 None。</p><h3>③ 共同更新：Actor Loss 与 Critic Loss</h3><p>先执行上面两组条件分支，再把 weights 和 critic_loss 交给下面的共享更新函数。Actor Loss 更新 Actor；Critic Loss 更新 Critic。两套网络参数独立，使用本轮更新前的预测。</p><pre class="update-code"></pre><p class="note">读法：returns（MC）或 target（TD）是希望 Critic 学到的值；v / q 是它现在的预测。critic_loss 衡量两者差距。backward() 计算梯度，step() 真正修改网络参数。方法 1、2 的 critic_loss = None，因此跳过 Critic 更新。</p>`;
 block.querySelector('.difference-code').textContent=methodCode[active];
 block.querySelector('.update-code').textContent=updateCode;
 block.querySelector('.critic-code').textContent=criticCode;
 block.querySelector('.setup-code').textContent=setupCode;
 $('method').appendChild(block);
}
'''
page = page.replace('let active=0;', extra + '\nlet active=0;')
page = page.replace('</style>', '.method-detail{margin-top:32px;border-top:1px solid var(--line);padding-top:26px}.method-detail h3{margin-top:24px}.method-detail td{white-space:normal}.training-flow{max-width:900px;margin:20px auto;text-align:center}.flow-node{background:#eaf1e9;border:1px solid var(--line);border-radius:12px;padding:16px;overflow-wrap:anywhere}.flow-arrow{padding:8px;color:var(--green);font-size:14px}.difference-code,.critic-code,.update-code{max-height:none}</style>')
notes = '''<div class="note"><b>本例的三个边界</b><p>方法 4、5 的 Q target 是完整 MC 回报 Gₜ；没有 bootstrap。若改为 on-policy Expected SARSA，可用 sg[rₜ + γ(1−dₜ)∑ₐ πθ(a|sₜ₊₁)Qω(sₜ₊₁,a)]，也可采样下一动作。DQN 则用 max 的最优控制 target；Q Critic 与 DQN 不是同一算法。</p><p>方法 6 的 target 只看一步，即使本框架为统一比较而收集完整 episode，也仍是一步 TD。dₜ 只表示真正终止；实现只在非终止 next state 上计算 V。一般外部截断不应置 d=1。本例三步就是任务终点；共享 rollout 拒绝不完整截断，避免把 MC 尾值错误置零。</p><p>所有权重取自本轮更新前的网络，采样期间参数固定。每轮 Actor 更新一次、所需 Critic 更新一次，然后重新采样；无 replay、无多轮复用旧轨迹。近似 Q / TD 的 Actor 梯度可能有偏差。</p></div>'''
page = page.replace('<section id="code"', notes + '<section id="code"')
(ROOT / 'index.html').write_text(page)
print('Built index.html from policy_gradient_demo.py')
