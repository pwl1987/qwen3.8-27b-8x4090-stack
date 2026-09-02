-- Qwen3.8-27B 动态会话粘滞负载均衡
-- 策略: ① 会话粘滞优先(key=请求体前512B, TTL 30min, 命中续期) → 前缀 KV 缓存命中最大化
--       ② 粘滞永不因池占用漂移(大会话漂移=每轮冷prefill更糟, 必须钉死原副本), 仅满载/不健康才漂
--       ③ 新会话 → 接入最闲健康副本; 池占用 ≥ POOL_HI 的副本不接新会话(大上下文会话独占)
--       ④ 大小会话分流(2026-09-01 v4): 请求体 >600KB(≈130K+ token) → 大会话池(r2/r3 独享,
--          每副本装 1 个 130K-256K 会话); 其余 → 小会话池(llama/r1)。避免大中小会话在共享
--          KV 池里互相驱逐挤撞(Context size exceeded / 反复冷 prefill 的根源)
--       ⑤ 副本状态探测缓存 2 秒; ⑥ 路由计数 /_lbstats; ⑦ DISABLED 软摘除(缩容 drain 用)
local SLOTS_PER       = 4      -- 每副本 slot 数(与 compose 参数一致)
local STICKY_TTL      = 1800   -- 会话映射 30 分钟
local STATE_TTL       = 2      -- 负载探测缓存 2 秒
local CL_HI           = 500000 -- 请求体字节阈值: 超过 → 大会话池(500KB≈110K+token, 给小池128K slot留18K裕量)

-- 大小分流 + 软摘除过滤
local cl = tonumber(ngx.req.get_headers()["content-length"] or 0) or 0
local backends = cl > CL_HI and {"r2:8080", "r3:8080"} or {"llama:8080", "r1:8080"}
local POOL_HI  = cl > CL_HI and 100000 or 131072  -- 大池: 已有≥100K驻留的副本不接新大会话(每副本一个); 小池 128K
do
  local df = io.open("/etc/nginx/lb/DISABLED", "r")
  if df then
    local disabled = {}
    for line in df:lines() do
      disabled[line:gsub("%s+", "")] = true
    end
    df:close()
    local kept = {}
    for _, b in ipairs(backends) do
      if not disabled[b] then kept[#kept + 1] = b end
    end
    if #kept > 0 then backends = kept end   -- 全部禁用时忽略文件, 防自锁
  end
end

local sessions = ngx.shared.sessions
local bstate   = ngx.shared.bstate
local lbstats  = ngx.shared.lbstats
local cjson    = require "cjson.safe"

-- ① 会话 key: 请求体前 512 字节(同一会话每轮相同; 超大 body 落盘时读文件兜底)
ngx.req.read_body()
local body = ngx.req.get_body_data()
if not body then
  local f = ngx.req.get_body_file()
  if f then
    local fh = io.open(f, "rb")
    if fh then
      body = fh:read(512) or ""
      fh:close()
    end
  end
end
local key = body and string.sub(body, 1, 512) or ""

-- ② 探测单个副本(带 2s 缓存): /slots 统计 busy 数 + 池占用(处理中任务的 prompt token 总和)
local function probe(b)
  local cached = bstate:get(b)
  if cached then
    return cjson.decode(cached) or {ok = false, busy = 99}
  end
  local res = ngx.location.capture("/_probe?b=" .. b)
  local st
  if res.status == 200 then
    local ok, slots = pcall(cjson.decode, res.body)
    local busy, total, pool = 0, 0, 0
    if ok and type(slots) == "table" then
      for _, s in ipairs(slots) do
        total = total + 1
        if s.is_processing then busy = busy + 1 end
        -- 池占用按所有 slot 的 prompt 估算(空闲 slot 保留上任务的驻留 KV, 一样占池)
        pool = pool + (s.n_prompt_tokens or 0)
      end
    end
    st = {ok = (ok and total > 0), busy = busy, pool = pool, nslots = total}
  else
    st = {ok = false, busy = 99, pool = 0}
  end
  bstate:set(b, cjson.encode(st), STATE_TTL)
  return st
end

local states = {}
for _, b in ipairs(backends) do states[b] = probe(b) end

-- ③ 选择: 粘滞优先(粘滞不看池——大会话漂移只会导致每轮冷prefill更糟, 必须钉死原副本);
--       池占用仅用于"新会话避让"(POOL_HI 已按大小池在顶部定义): 池高的副本不接新会话
local sticky = sessions:get(key)
local target
if sticky and states[sticky] and states[sticky].ok
        and states[sticky].busy < (states[sticky].nslots or SLOTS_PER) then
  target = sticky
  lbstats:incr("sticky_hit", 1, 0)
  sessions:expire(key, STICKY_TTL)   -- 活跃会话续期, 防止长会话中途过期被重分配
else
  if sticky then
    lbstats:incr("drift", 1, 0)      -- 有映射但目标满载/不健康 → 漂移
  else
    lbstats:incr("new_sess", 1, 0)   -- 新会话接入
  end
  local healthy = {}
  for _, b in ipairs(backends) do
    if states[b].ok and (states[b].pool or 0) < POOL_HI then healthy[#healthy + 1] = b end
  end
  if #healthy == 0 then
    -- 所有副本池占用都高: 退回仅健康检查(不再看池), 让请求进最少占用的副本排队
    for _, b in ipairs(backends) do
      if states[b].ok then healthy[#healthy + 1] = b end
    end
  end
  if #healthy == 0 then
    target = backends[1]   -- 全不健康也放行, 让 proxy 层暴露真实错误
  else
    -- 最小 busy 的候选里随机挑(平手随机化, 避免全空闲时永远打第一个)
    local min_busy, cands = math.huge, {}
    for _, b in ipairs(healthy) do
      local bu = states[b].busy
      if bu < min_busy then min_busy, cands = bu, {b}
      elseif bu == min_busy then cands[#cands + 1] = b end
    end
    target = cands[math.random(#cands)]
  end
  sessions:set(key, target, STICKY_TTL)
end
lbstats:incr("sel_" .. target, 1, 0)

ngx.var.backend = target
