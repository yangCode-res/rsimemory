"""Read-only experiment summaries; full clinical trajectories stay in Langfuse."""
import json
from pathlib import Path
from urllib.parse import urlencode


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def snapshot(root):
    from scripts.evolution_overview import diagnostic_evidence, search_state, comparison
    from scripts.langfuse_run import digest
    root = Path(root)
    runs = []
    for directory in sorted((root / 'runs').glob('full-v03-local-*'), reverse=True):
        result = directory / 'result'
        config = load(result / 'config.json')
        summary = load(result / 'summary.json')
        if not config or config.get('fixture') or not summary:
            continue
        tree = load(result / 'tree.json')
        state = load(result / 'run-status.json')
        diagnosis = diagnostic_evidence(result)
        links = {}
        # Only verified published overview projections are advertised as drill-down links.
        for index in sorted((root / 'runs').glob('langfuse-import-' + directory.name.split('-')[-1] + '*/evolution-overview-*/index.json')):
            if load(index.with_name('verification.json')).get('verified'):
                value = load(index)
                links['overview'] = 'http://localhost:3000' + value.get('url', '')
                links['trace_id'] = value.get('trace_id')
        policies = []
        branches = summary.get('branches', [])
        specs = [{'id':'plain-bm25','parent':None}, {'id':'plain-semantic','parent':None}] + branches
        for branch in specs:
            policy = branch['id']; parent = branch.get('parent') or 'plain-bm25'
            proposal = branch.get('proposal') or {}
            folds = {}
            for fold in ('train', 'validation'):
                base = result / ('branches/' + policy if policy.startswith('candidate-') else policy) / fold
                report = load(base / 'report.json')
                progress = load(base / 'progress.json')
                parent_report = load(result / parent / fold / 'report.json')
                folds[fold] = {'report_available': bool(report), 'progress': progress,
                    **comparison(report, parent_report),
                    'policy_seconds':report.get('policy_seconds', summary.get('comparison', {}).get(policy, {}).get('policy_seconds')) if fold == 'validation' else report.get('policy_seconds'),
                    'policy_bytes':report.get('policy_bytes', summary.get('comparison', {}).get(policy, {}).get('policy_bytes')) if fold == 'validation' else report.get('policy_bytes')}
            policies.append({'id':policy, 'parent':branch.get('parent'), 'commit':branch.get('commit'),
                'status':branch.get('status'), 'direction':proposal.get('direction'),
                'detail_url':links.get('overview','') + '?observation=' + digest([links['trace_id'],policy])[:16] if links.get('trace_id') and policy.startswith('candidate-') else None,
                'prediction':proposal.get('prediction'), 'risk':proposal.get('risk'),
                'target_clusters':proposal.get('supporting_cluster_labels', []),
                'folds':folds, 'search':search_state(tree, policy)})
        # Preserve aggregate evidence, not clinical text or raw persona identifiers.
        clusters = []
        for cluster_index, cluster in enumerate(diagnosis.pop('clusters', [])):
            ids = cluster.get('supporting_case_ids', cluster.get('case_ids', []))
            label = cluster.get('label')
            clusters.append({'label':label, 'cited_cases':len(ids),
                'detail_url':links.get('overview','') + '?observation=' + digest([links['trace_id'],'cluster',cluster_index])[:16] if links.get('trace_id') else None,
                'attribution_status':cluster.get('attribution_status'),
                'uncertainty':cluster.get('uncertainty'),
                'candidates':[p['id'] for p in policies if label in p['target_clusters']]})
        links['patients'] = 'http://localhost:3000/project/memory-policy-rsi/sessions?' + urlencode({
            'dateRange':'1y','filter':'id;string;;contains;' + directory.name + '/patient-'})
        runs.append({'id':directory.name, 'state':state.get('status',summary.get('status','historical_archive')),
            'stage':state.get('stage'), 'test_evaluated':summary.get('test_evaluated',False),
            'framework_commit':load(result/'config.json').get('implementation_commit'),
            'baseline_reuse':bool(load(result/'baseline-reuse.json').get('source_manifest_verified')),
            'policies':policies, 'diagnostics':diagnosis, 'clusters':clusters, 'links':links})
    return {'runs':runs, 'source':'server archived reports; not a live Langfuse ingestion feed'}


HTML = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RSI 进化观测</title>
<style>
body{margin:0;background:#f4f6fa;color:#182336;font:15px system-ui,'Microsoft YaHei',sans-serif}main{max-width:1450px;margin:auto;padding:28px}h1{font-size:26px;margin:0 0 8px}h2{font-size:19px;margin:28px 0 12px}p{line-height:1.7}.muted{color:#667489}.bar,.cards{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.card{background:white;border:1px solid #dde3ed;border-radius:12px;padding:16px;flex:1;min-width:160px}.card strong{font-size:22px;display:block;margin-top:7px}select,button,a.button{border:1px solid #cbd5e1;background:white;border-radius:8px;padding:9px 12px;color:#254d86;text-decoration:none;cursor:pointer}select{max-width:100%}.panel{background:white;border:1px solid #dde3ed;border-radius:12px;padding:18px;margin-top:16px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:12px 10px;border-bottom:1px solid #e6eaf0;vertical-align:top}th{color:#536278;background:#f8fafc;white-space:nowrap}td.direction{min-width:240px;max-width:390px;line-height:1.6}.positive{color:#12734c}.negative{color:#b13b42}.badge{display:inline-block;background:#eaf0fb;color:#355b91;padding:3px 8px;border-radius:5px;font-size:12px;margin:3px}summary{cursor:pointer;font-weight:600}details p{margin:8px 0}.small{font-size:12px}.num{white-space:nowrap}a{color:#285cb0}#error{color:#b13b42}
</style><main><h1>RSI 进化观测</h1><p class="muted">先看轮次与策略结果，再按错误类型、患者和问题深入轨迹。正式实验与合成测试分开。</p>
<div class="bar"><select id="run"></select><button id="refresh">立即刷新</button><label><input id="auto" type="checkbox" checked>每 30 秒刷新</label><span id="updated" class="muted small"></span></div><p id="error"></p><div id="content"></div></main>
<script>
let data=null;const $=id=>document.getElementById(id);const esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=x=>typeof x==='number'?(100*x).toFixed(2)+'%':'—';const num=x=>typeof x==='number'?x.toFixed(3):'—';
function delta(x){return typeof x==='number'?`<span class="${x>0?'positive':x<0?'negative':''}">${x>0?'+':''}${x.toFixed(2)} pp</span>`:'—'}
const status={running:'运行中',completed:'已完成',failed:'失败',evaluated:'评分完整',incomplete_scoring:'评分不完整',pending:'待合并',historical_archive:'历史归档',no_fully_scored_candidates:'无完整评分候选'};
function fold(f){if(!f.report_available){let p=f.progress||{};return `<span class="muted">${esc(status[p.status]||p.status||'未开始／暂无报告')} ${p.total_questions?esc(p.completed)+' / '+esc(p.total_questions):''}</span>`}return `<b>${pct(f.E2E_AVG)}</b><br><span class="small muted">严格 AVG ${pct(f.AVG)} · 异常 ${pct(f.abnormal_rate)}</span>`}
function render(){let r=data.runs.find(x=>x.id===$('run').value);if(!r){$('content').innerHTML='<p>暂无正式实验报告。</p>';return}let cs=r.policies.filter(p=>p.id.startsWith('candidate-'));let best=r.policies.filter(p=>typeof p.folds.validation.E2E_AVG==='number').sort((a,b)=>b.folds.validation.E2E_AVG-a.folds.validation.E2E_AVG)[0];let d=r.diagnostics.observed_groups||{};
$('content').innerHTML=`<div class="cards"><div class="card">当前状态<strong>${esc(status[r.state]||r.state)}</strong><span class="small muted">${esc(r.stage||'以正式报告为准')}</span></div><div class="card">候选策略<strong>${cs.length}</strong><span class="small muted">${cs.filter(p=>p.folds.validation.report_available).length} 个已有验证报告</span></div><div class="card">当前验证最高<strong>${best?pct(best.folds.validation.E2E_AVG):'暂无结果'}</strong><span class="small muted">${esc(best?.id||'尚未评测完成')}；单次结果，不代表稳定优势</span></div><div class="card">轨迹导入<strong>${r.links.overview?'总览已核验':'未确认'}</strong><span class="small muted">进度刷新与 Langfuse 导入独立</span></div></div>
<div class="panel"><b>阅读顺序</b>：提议方向 → 训练／验证变化 → 错误与题型 → 搜索状态 → 详细轨迹。<p class="small muted">表中主成绩为 E2E AVG；Δ 相对父策略，单位为百分点。缺失不计零；验证集不参与训练奖励回传。测试集：${r.test_evaluated?'已评测':'未评测'}。${r.baseline_reuse?'本轮基线复用自已校验的历史运行，候选单独重新评测。':''}</p>
<div class="bar">${r.links.overview?`<a class="button" target="_blank" href="${esc(r.links.overview)}">深入：Langfuse 轮次与提案</a>`:'<span class="muted">此轮还没有经过核验的 Langfuse 总览链接</span>'}<a class="button" target="_blank" href="${esc(r.links.patients)}">深入：患者 Sessions（仅已导入数据）</a><a target="_blank" href="http://localhost:3000/project/memory-policy-rsi/traces">全部底层轨迹</a></div></div>
<h2>策略评测对照</h2><div class="panel scroll"><table><thead><tr><th>策略／父版本</th><th>提议方向</th><th>训练 E2E AVG</th><th>训练 Δ</th><th>验证 E2E AVG</th><th>验证 Δ</th><th>搜索 Q／N／UCT</th></tr></thead><tbody>${r.policies.map(p=>{let q=p.search;return `<tr><td><b>${p.detail_url?`<a target="_blank" href="${esc(p.detail_url)}">${esc(p.id)} ↗</a>`:esc(p.id)}</b><br><span class="small muted">父：${esc(p.parent)}<br>${esc(status[p.status]||p.status||'基线')}<br>${esc(p.commit?.slice(0,8))}</span></td><td class="direction">${esc(p.direction||(p.id==='plain-bm25'?'BM25 固定对照':'语义检索固定对照'))}${p.target_clusters.length?'<br>'+p.target_clusters.map(x=>`<span class="badge">${esc(x)}</span>`).join(''):''}</td><td class="num">${fold(p.folds.train)}</td><td class="num">${p.parent?delta(p.folds.train.delta_vs_parent_pp?.E2E_AVG):'—'}</td><td class="num">${fold(p.folds.validation)}</td><td class="num">${p.parent?delta(p.folds.validation.delta_vs_parent_pp?.E2E_AVG):'—'}</td><td class="small num">Q ${num(q.Q)} / N ${q.visits??'—'}<br>UCT ${q.priority==='unvisited_positive_infinity'?'未访问 ∞':q.priority==='excluded_negative_infinity'?'排除 −∞':num(q.uct)}<br><span class="muted">树状态快照，非历史选择</span></td></tr>`}).join('')}</tbody></table></div>
<h2>错误诊断：全量观测与模型归因分开看</h2><div class="panel"><p>训练题数 <b>${esc(d.total_questions)}</b> · 答对 ${esc(d.success_count)} · 答错 ${esc(d.failure_count)} · 异常 ${esc(d.abnormal_count)} · 诊断样本 ${esc(r.diagnostics.sample_coverage?.total)}</p><p class="small muted">“—”表示历史档案缺少对应字段；模型聚类的引用案例数不是全量发生率，检索命中代理也不是因果结论。</p><div class="bar">${Object.entries(d.error_groups||{}).map(([k,v])=>`<span class="badge">${esc(({abnormal_actor_failure_observed:'回答异常',forced_final_observed:'触发强制作答',gold_session_hit_proxy:'命中参考会话代理',gold_session_miss_proxy:'未命中参考会话代理',no_read_observed:'未读取证据',success:'答对',tool_error_observed:'工具报错'})[k]||k)}：${esc(v)}</span>`).join('')}</div><table><thead><tr><th>模型报告的错误聚类</th><th>引用案例</th><th>归因状态</th><th>对应候选</th></tr></thead><tbody>${r.clusters.map(c=>`<tr><td>${c.detail_url?`<a target="_blank" href="${esc(c.detail_url)}">${esc(c.label)} ↗</a>`:esc(c.label)}</td><td>${c.cited_cases}</td><td>${esc(c.attribution_status||'历史模型归因；未裁定')}</td><td>${esc(c.candidates.join('、')||'无明确关联')}</td></tr>`).join('')}</tbody></table></div>
<h2>按需展开：预测、风险、题型与成本</h2>${cs.map(p=>`<details class="panel"><summary>${esc(p.id)}：详细对照</summary><p><b>模型预测：</b>${esc(p.prediction)}</p><p><b>模型自报风险：</b>${esc(p.risk)}</p><div class="scroll"><table><tr><th>题型</th><th>训练 E2E</th><th>训练 Δ</th><th>验证 E2E</th><th>验证 Δ</th></tr>${[...new Set([...Object.keys(p.folds.train.e2e_type_accuracy||{}),...Object.keys(p.folds.validation.e2e_type_accuracy||{})])].map(k=>`<tr><td>${esc(k)}</td><td>${pct(p.folds.train.e2e_type_accuracy?.[k])}</td><td>${delta(p.folds.train.type_delta_vs_parent_pp?.[k])}</td><td>${pct(p.folds.validation.e2e_type_accuracy?.[k])}</td><td>${delta(p.folds.validation.type_delta_vs_parent_pp?.[k])}</td></tr>`).join('')}</table></div><p class="small muted">验证策略耗时 ${num(p.folds.validation.policy_seconds)} 秒；状态体积 ${typeof p.folds.validation.policy_bytes==='number'?(p.folds.validation.policy_bytes/1048576).toFixed(2)+' MiB':'—'}。包含隔离等开销，不等于纯算法成本。预测尚未自动作因果判定。</p></details>`).join('')}`;}
async function refresh(){try{const res=await fetch('/api/overview',{cache:'no-store'});if(!res.ok)throw Error('HTTP '+res.status);data=await res.json();const old=$('run').value;$('run').innerHTML=data.runs.map(r=>`<option value="${esc(r.id)}">${esc(r.id)} · ${esc(status[r.state]||r.state)}</option>`).join('');if(data.runs.some(r=>r.id===old))$('run').value=old;render();$('updated').textContent='读取时间 '+new Date().toLocaleTimeString();$('error').textContent='';}catch(e){$('error').textContent='读取失败，保留上次显示：'+e.message}}
$('run').onchange=render;$('refresh').onclick=refresh;setInterval(()=>{if($('auto').checked)refresh()},30000);refresh();
</script></html>"""
