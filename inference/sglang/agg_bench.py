#!/usr/bin/env python3
import argparse, json, time, concurrent.futures, urllib.request
def post(url, payload, timeout=900):
    d=json.dumps(payload).encode()
    r=urllib.request.Request(url,data=d,headers={"Content-Type":"application/json"})
    return urllib.request.urlopen(r,timeout=timeout)
def one(base,model,n):
    p={"model":model,"messages":[{"role":"user","content":"Write a short paragraph about the sea."}],"max_tokens":n,"temperature":0}
    t0=time.time()
    try:
        b=json.loads(post(base+"/v1/chat/completions",p).read().decode())
        dt=time.time()-t0
        tok=(b.get("usage") or {}).get("completion_tokens", n)
        return dict(ok=True,tok=tok,lat=dt)
    except Exception as e:
        return dict(ok=False,err=repr(e)[:100],lat=time.time()-t0)
def run(base,model,c,n=128):
    t_start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
        res=[f.result() for f in [ex.submit(one,base,model,n) for _ in range(c)]]
    wall=time.time()-t_start
    ok=[r for r in res if r["ok"]]
    tot=sum(r["tok"] for r in ok)
    return dict(concurrency=c,n_req=len(res),n_ok=len(ok),wall_s=round(wall,2),
                total_tok=tot,agg_tps=round(tot/wall,1) if wall>0 else 0,
                per_req_tps_mean=round(sum(r["tok"]/r["lat"] for r in ok)/len(ok),1) if ok else 0,
                fails=[r["err"] for r in res if not r["ok"]][:2])
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True); ap.add_argument("--model",default="qwen3.8-27b")
    ap.add_argument("--conc",default="8,16,24")
    a=ap.parse_args()
    out={}
    for c in a.conc.split(","):
        c=int(c)
        out[f"c{c}"]=run(a.base,a.model,c)
        print(json.dumps(out[f"c{c}"]),flush=True)
    json.dump(out,open(a.base.split("//")[-1].split(":")[0]+f"_agg_{a.model}.json","w"))
if __name__=="__main__": main()
