#!/usr/bin/env python3
# 监控采集 v3: v2(GPU+副本指标+每路速率+宿主系统) + 24h 历史(10s 粒度, 落盘 ./mon/hist 跨重启保留)
#   + QPS/累计请求数(/slots 的 id_task 是副本内单调任务号, 差分即请求速率)
#   + TTFT 探测(热: 固定 prompt 走缓存命中路径 30s 一次; 冷: 唯一短 prompt 5min 一次, 经 LB :8000)
#   + LB 路由统计(/_lbstats: 粘滞命中/漂移/新会话/每副本选中)
import json, os, re, shutil, socket, subprocess, threading, time, urllib.request
from collections import deque
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

BACKENDS = [('llama', 'llama', 8081), ('r1', 'r1', 8082), ('r2', 'r2', 8083), ('r3', 'r3', 8084)]
# 通用服务健康探活(compose 环境变量 MON_SERVICES="名字=URL,..." 同 docker 网可达), 如 ComfyUI=http://comfyui:8188/system_stats
SERVICES = [tuple(x.split('=', 1)) for x in os.environ.get('MON_SERVICES', '').split(',') if '=' in x]
WWW, DATA = '/www', '/www/data.json'
HIST_FILE = '/www/hist/history.json'          # compose 挂载 ./mon/hist, 跨容器重建保留
INTERVAL, HIST_STEP = 2, 10                    # 采样 2s; 历史聚合 10s
HIST_KEEP = 24 * 3600 // HIST_STEP             # 8640 点 = 24h
FIELDS = ['index', 'name', 'utilization.gpu', 'memory.used', 'memory.total', 'temperature.gpu', 'power.draw']
LB = 'http://lb:8000'

def gpus():
    # nvidia-smi 偶发失败不能拖垮整个 snapshot(否则 data.json 停更, 页面冻结在旧数据)
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(FIELDS),
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
    except Exception:
        return []
    gs = []
    for line in out.stdout.strip().splitlines():
        p = [x.strip() for x in line.split(',')]
        if len(p) < 7:
            continue
        gs.append({'index': int(p[0]), 'name': p[1], 'utilization_gpu': int(p[2]),
                   'memory_used': int(p[3]), 'memory_total': int(p[4]),
                   'temperature_gpu': int(p[5]), 'power_w': round(float(p[6]))})
    return gs

M_KEYS = ('prompt_tokens_total', 'prompt_tokens_cached_total', 'tokens_predicted_total',
          'spec_decode_num_draft_tokens_total', 'spec_decode_num_accepted_tokens_total',
          'requests_processing', 'requests_deferred')

# ---------- GPU 进程归属: compute-apps + /host/proc cgroup 容器id → docker.sock 反查容器名 ----------

def http_unix(sock_path, path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect(sock_path)
        s.sendall(f'GET {path} HTTP/1.0\r\nHost: docker\r\n\r\n'.encode())
        buf = b''
        while True:
            c = s.recv(65536)
            if not c:
                break
            buf += c
    finally:
        s.close()
    return buf.split(b'\r\n\r\n', 1)[1] if b'\r\n\r\n' in buf else b''

_pidname_cache = {}

def pid_name(pid):
    if pid in _pidname_cache:
        return _pidname_cache[pid]
    name = f'pid {pid}'
    cid = None
    try:
        with open(f'/host/proc/{pid}/cgroup') as f:
            # cgroup v2 容器内视角: "/../docker-<id>.scope"; 宿主视角: "/docker/<id>"
            m = re.search(r'docker[-/]([0-9a-f]{64})', f.read())
            if m:
                cid = m.group(1)
    except Exception:
        pass
    if cid:
        try:
            info = json.loads(http_unix('/var/run/docker.sock', f'/containers/{cid}/json'))
            n = (info.get('Name') or '').strip('/')
            if n:
                name = n
        except Exception:
            pass
    if name == f'pid {pid}':
        try:
            with open(f'/host/proc/{pid}/cmdline', 'rb') as f:
                cl = f.read().split(b'\0')
            if cl and cl[0]:
                name = os.path.basename(cl[0].decode(errors='replace'))[:24]
        except Exception:
            pass
    _pidname_cache[pid] = name
    return name

def gpu_procs():
    # 每张卡上是谁在算(容器名/进程名 + 占用显存); GPU0-3 应恒为 4 副本, GPU4-7 出现进程即有新负载
    try:
        ls = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5).stdout
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    uuid_idx = {}
    for line in ls.splitlines():
        m = re.match(r'GPU (\d+): .*(GPU-[0-9a-f-]+)', line)
        if m:
            uuid_idx[m.group(2)] = int(m.group(1))
    res = []
    for line in apps.strip().splitlines():
        p = [x.strip() for x in line.split(',')]
        if len(p) < 3:
            continue
        gpu = uuid_idx.get(p[0])
        try:
            pid, mem = int(p[1]), int(p[2])
        except ValueError:
            continue
        if gpu is None:
            continue
        res.append({'gpu': gpu, 'pid': pid, 'mem': mem, 'name': pid_name(pid)})
    res.sort(key=lambda x: x['gpu'])
    return res

# ---------- 通用服务探活(ComfyUI /system_stats 等) ----------

def probe_services():
    out = []
    for name, url in SERVICES:
        rec = {'name': name, 'ok': False, 'ms': None, 'info': None}
        try:
            t0 = time.time()
            with urllib.request.urlopen(url, timeout=3) as r:
                body = r.read(65536)
                rec['ok'] = True
                rec['ms'] = round((time.time() - t0) * 1000)
                try:
                    j = json.loads(body)
                    if isinstance(j, dict) and j.get('devices'):    # ComfyUI /system_stats 格式
                        dev = j['devices'][0]
                        vt, vf = dev.get('vram_total', 0), dev.get('vram_free', 0)
                        if vt:
                            rec['info'] = f"显存空闲 {vf/2**30:.0f}/{vt/2**30:.0f}G"
                except Exception:
                    pass
        except Exception as e:
            rec['info'] = type(e).__name__
        out.append(rec)
    return out


def fetch_metrics(host):
    with urllib.request.urlopen(f'http://{host}:8080/metrics', timeout=3) as r:
        t = r.read().decode()
    out = {}
    for k in M_KEYS:
        mm = re.search(rf'^llamacpp:{k}(?:\{{[^}}]*\}})? (\d+(?:\.\d+)?)', t, re.M)
        out[k] = float(mm.group(1)) if mm else 0.0
    return out

def fetch_slots(host):
    with urllib.request.urlopen(f'http://{host}:8080/slots', timeout=3) as r:
        s = json.load(r)
    out = []
    for x in s:
        nt = (x.get('next_token') or [{}])[0]
        out.append({'id': x.get('id'), 'busy': bool(x.get('is_processing')),
                    'prompt_tok': x.get('n_prompt_tokens', 0),
                    'cache_tok': x.get('n_prompt_tokens_cache', 0),
                    'n_decoded': nt.get('n_decoded', 0) if x.get('is_processing') else 0,
                    'id_task': x.get('id_task', 0)})
    return out

def fetch_lb():
    with urllib.request.urlopen(LB + '/_lbstats', timeout=2) as r:
        return json.load(r)

def sysinfo(prev_cpu):
    # 宿主 CPU%/内存/负载(/host/proc 由 compose 挂载); 返回 (info, 新的累计cpu样本)
    def read_cpu():
        with open('/host/proc/stat') as f:
            p = f.readline().split()[1:]
        v = [int(x) for x in p]
        return sum(v), v[3] + v[4]          # total, idle(+iowait)
    mem, load = {}, [0, 0, 0]
    try:
        with open('/host/proc/meminfo') as f:
            mem = {l.split(':')[0]: int(l.split()[1]) for l in f if ':' in l}
        load = [float(x) for x in open('/host/proc/loadavg').read().split()[:3]]
    except Exception as e:
        try:
            with open(os.path.join(WWW, 'err.txt'), 'a') as f:
                f.write(f'{time.ctime()} sysinfo: {type(e).__name__}: {e}\n')
        except Exception:
            pass
    total, idle = read_cpu()
    cpu_pct = 0.0
    if prev_cpu:
        dt, di = total - prev_cpu[0], idle - prev_cpu[1]
        cpu_pct = max(0.0, min(100.0, 100 * (1 - di / dt))) if dt > 0 else 0.0
    si = {'cpu_pct': round(cpu_pct, 1),
          'mem_used': (mem.get('MemTotal', 0) - mem.get('MemAvailable', 0)) * 1024,   # 统一输出 bytes
          'mem_total': mem.get('MemTotal', 0) * 1024, 'load1': load[0]}
    return si, (total, idle)

state = {'prev_rep': {}, 'prev_cpu': None, 'prev_slot': {}, 'prev_task': {},   # prev_slot: rep -> {slot_id: (ts, n_decoded)}
         'ttft': {'hot': None, 'hot_ts': 0, 'cold': None, 'cold_ts': 0}, 'qps_ema': 0.0}

def slot_rates(key, slots, now):
    # per-slot 实时速率: busy slot 的 n_decoded 差分(注意 counter 只在请求结束才累计进 metrics,
    # n_decoded 是任务内实时值, 任务结束归零——差分只对连续 busy 的 slot 有效)
    prev = state['prev_slot'].setdefault(key, {})
    rates, cur = [], {}
    for s in slots:
        if not s['busy']:
            continue
        p = prev.get(s['id'])
        if p and now - p[0] > INTERVAL * 0.5 and s['n_decoded'] >= p[1]:
            rates.append(max(0.0, (s['n_decoded'] - p[1]) / (now - p[0])))
        cur[s['id']] = (now, s['n_decoded'])
    state['prev_slot'][key] = cur
    return rates

def task_rates(key, slots, now):
    # id_task 是 llama-server 副本内单调任务号, 各 slot 取 max 即"已受理任务数"
    task = max((s['id_task'] for s in slots), default=None)
    p = state['prev_task'].get(key)
    qps = None
    if task is not None and p and now > p[0]:
        qps = round(max(0.0, (task - p[1]) / (now - p[0])), 2)
    if task is not None:
        state['prev_task'][key] = (now, task)
    return qps, (task if task is not None else (p[1] if p else 0))

def snapshot():
    now = time.time()
    reps, tot = {}, {'gen_tps': 0.0, 'pre_tps': 0.0, 'processing': 0, 'deferred': 0,
                     'prompt_total': 0.0, 'cached_total': 0.0, 'draft': 0.0, 'accepted': 0.0,
                     'qps': 0.0, 'req_total': 0}
    for key, host, port in BACKENDS:
        try:
            mm = fetch_metrics(host)
            sl = fetch_slots(host)
            p = state['prev_rep'].get(key)
            dt = now - p[0] if p else 0
            if p and dt > INTERVAL * 0.5:
                pre = max(0.0, (mm['prompt_tokens_total'] - p[1]['prompt_tokens_total']) / dt)
            else:
                pre = 0.0
            state['prev_rep'][key] = (now, mm)
            stream_tps = slot_rates(key, sl, now)
            gen = sum(stream_tps)
            qps, req_total = task_rates(key, sl, now)
            reps[key] = {'ok': True, 'gen_tps': round(gen, 1), 'pre_tps': round(pre, 0),
                         'stream_tps': [round(x, 1) for x in stream_tps],
                         'processing': int(mm['requests_processing']), 'deferred': int(mm['requests_deferred']),
                         'slots': sl, 'qps': qps, 'req_total': req_total,
                         'cache_hit': round(100 * mm['prompt_tokens_cached_total'] / mm['prompt_tokens_total'], 1) if mm['prompt_tokens_total'] else 0.0,
                         'mtp_acc': round(100 * mm['spec_decode_num_accepted_tokens_total'] / mm['spec_decode_num_draft_tokens_total'], 1) if mm['spec_decode_num_draft_tokens_total'] else 0.0}
            tot['gen_tps'] += gen; tot['pre_tps'] += pre
            tot['processing'] += int(mm['requests_processing']); tot['deferred'] += int(mm['requests_deferred'])
            tot['prompt_total'] += mm['prompt_tokens_total']; tot['cached_total'] += mm['prompt_tokens_cached_total']
            tot['draft'] += mm['spec_decode_num_draft_tokens_total']; tot['accepted'] += mm['spec_decode_num_accepted_tokens_total']
            if qps is not None:
                tot['qps'] += qps
            tot['req_total'] += req_total
        except Exception:
            reps[key] = {'ok': False, 'gen_tps': 0, 'pre_tps': 0, 'processing': 0, 'deferred': 0,
                         'slots': [], 'qps': None, 'req_total': 0}
    tot['gen_tps'] = round(tot['gen_tps'], 1); tot['pre_tps'] = round(tot['pre_tps'], 0)
    tot['qps'] = round(tot['qps'], 2)
    state['qps_ema'] = round(0.6 * state['qps_ema'] + 0.4 * tot['qps'], 2)   # 瞬时差分抖动大, KPI 显示平滑值
    tot['qps_ema'] = state['qps_ema']
    tot['cache_hit'] = round(100 * tot['cached_total'] / tot['prompt_total'], 1) if tot['prompt_total'] else 0.0
    tot['mtp_acc'] = round(100 * tot['accepted'] / tot['draft'], 1) if tot['draft'] else 0.0
    si, state['prev_cpu'] = sysinfo(state['prev_cpu'])
    try:
        lb = fetch_lb()
    except Exception:
        lb = None
    t = state['ttft']
    return {'ts': now, 'gpus': gpus(), 'reps': reps, 'total': tot, 'sys': si,
            'lb': lb, 'procs': gpu_procs(), 'services': probe_services(),
            'ttft': {'hot': t['hot'], 'hot_ts': t['hot_ts'],
                     'cold': t['cold'], 'cold_ts': t['cold_ts']}}

# ---------- 24h 历史: 10s 聚合一条, deque 环形, 每分钟原子落盘 ----------

def make_acc():
    return {'n': 0, 'gen': 0.0, 'pre': 0.0, 'conc': 0, 'def': 0, 'cpu': 0.0,
            'cache': 0.0, 'mtp': 0.0, 'qps': None, 'mem': 0.0,
            'ttft_h': None, 'ttft_c': None, 'g': []}

def hist_push(d):
    a, n = state['acc'], 0
    a['n'] += 1
    a['gen'] += d['total']['gen_tps']; a['pre'] += d['total']['pre_tps']
    a['conc'] = max(a['conc'], d['total']['processing'])
    a['def'] = max(a['def'], d['total']['deferred'])
    a['cpu'] += d['sys']['cpu_pct']
    a['cache'], a['mtp'] = d['total']['cache_hit'], d['total']['mtp_acc']
    a['qps'] = d['total']['qps']
    a['mem'] = round(100 * d['sys']['mem_used'] / d['sys']['mem_total'], 1) if d['sys']['mem_total'] else 0
    a['ttft_h'], a['ttft_c'] = d['ttft']['hot'], d['ttft']['cold']
    gs = a['g']
    while len(gs) < len(d['gpus']):
        gs.append([0, 0, 0.0, 0, 0])      # [util_sum, n, mem_pct, temp_max, power_sum]
    for i, g in enumerate(d['gpus']):
        b = gs[i]
        b[0] += g['utilization_gpu']; b[1] += 1
        b[2] = round(100 * g['memory_used'] / g['memory_total'], 1)
        b[3] = max(b[3], g['temperature_gpu'])
        b[4] += g['power_w']

def hist_flush(ts):
    a = state['acc']
    if not a['n']:
        return
    g = [[round(b[0] / b[1]), b[2], b[3], round(b[4] / b[1])] if b[1] else [0, 0, 0, 0] for b in a['g']]
    state['hist'].append({
        'ts': int(ts), 'gen': round(a['gen'] / a['n'], 1), 'pre': round(a['pre'] / a['n']),
        'conc': a['conc'], 'def': a['def'], 'cache': a['cache'], 'mtp': a['mtp'],
        'qps': a['qps'], 'cpu': round(a['cpu'] / a['n'], 1), 'mem': a['mem'],
        'th': a['ttft_h'], 'tc': a['ttft_c'], 'g': g})
    state['acc'] = make_acc()

def hist_load():
    h = deque(maxlen=HIST_KEEP)
    try:
        with open(HIST_FILE) as f:
            cut = time.time() - 24 * 3600
            for p in json.load(f).get('points', []):
                if p.get('ts', 0) >= cut:
                    h.append(p)
    except Exception:
        pass
    return h

def hist_persist():
    tmp = HIST_FILE + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump({'step': HIST_STEP, 'points': list(state['hist'])}, f, separators=(',', ':'))
        os.replace(tmp, HIST_FILE)
    except Exception as e:
        try:
            with open(os.path.join(WWW, 'err.txt'), 'a') as f:
                f.write(f'{time.ctime()} persist: {type(e).__name__}: {e}\n')
        except Exception:
            pass

# ---------- TTFT 探测(经 LB, 流式首块时间; max_tokens=1 生成开销可忽略) ----------

def ttft_probe(content):
    body = json.dumps({'messages': [{'role': 'user', 'content': content}],
                       'stream': True, 'max_tokens': 1}).encode()
    req = urllib.request.Request(LB + '/v1/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=20) as r:
        for line in r:
            if line.startswith(b'data:'):
                s = line[5:].strip()
                if s and s != b'[DONE]':
                    return round(time.time() - t0, 3)
    return None

def ttft_loop(content_fn, interval, key):
    while True:
        try:
            v = ttft_probe(content_fn())
            if v is not None:
                state['ttft'][key] = v
                state['ttft'][key + '_ts'] = time.time()
        except Exception:
            pass                                    # 探测失败不清旧值, UI 显示最近一次
        time.sleep(interval)

def loop():
    state['hist'], state['acc'] = hist_load(), make_acc()
    last_flush, last_persist = 0, 0
    while True:
        try:
            now = time.time()
            d = snapshot()
            hist_push(d)
            with open(DATA + '.tmp', 'w') as f:
                json.dump(d, f)
            os.replace(DATA + '.tmp', DATA)
            if now - last_flush >= HIST_STEP:
                hist_flush(int(now // HIST_STEP) * HIST_STEP)
                last_flush = now
            if now - last_persist >= 60:
                hist_persist()
                last_persist = now
        except Exception as e:
            try:
                with open(os.path.join(WWW, 'err.txt'), 'a') as f:
                    f.write(f'{time.ctime()} {type(e).__name__}: {e}\n')
            except Exception:
                pass
        time.sleep(INTERVAL)

os.makedirs(os.path.dirname(HIST_FILE), exist_ok=True)
os.makedirs(WWW, exist_ok=True)
shutil.copy('/app/index.html', os.path.join(WWW, 'index.html'))
threading.Thread(target=loop, daemon=True).start()
threading.Thread(target=ttft_loop, daemon=True,
                 kwargs={'content_fn': lambda: '监控健康探测, 固定前缀以命中缓存。请只回复 ok。',
                         'interval': 30, 'key': 'hot'}).start()
threading.Thread(target=ttft_loop, daemon=True,
                 kwargs={'content_fn': lambda: f'冷启动探测 {time.time()}, 内容唯一不走缓存。请只回复 ok。',
                         'interval': 300, 'key': 'cold'}).start()
os.chdir(WWW)
SimpleHTTPRequestHandler.log_message = lambda *a: None
ThreadingHTTPServer(("0.0.0.0", 8080), SimpleHTTPRequestHandler).serve_forever()
