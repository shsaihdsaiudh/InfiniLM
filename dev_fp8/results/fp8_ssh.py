#!/usr/bin/env python3
"""FP8 评测服务器命令通道(Gitee AI 容器,密码认证,pubkey 不可用)。

用法:
    python3 dev_fp8/fp8_ssh.py run 'nvidia-smi'
    python3 dev_fp8/fp8_ssh.py push ./local_file /root/fp8/
    python3 dev_fp8/fp8_ssh.py pull /root/fp8/x.csv dev_fp8/results/
    python3 dev_fp8/fp8_ssh.py bg <name> '<shell 命令>'   # tmux 后台,日志 /root/fp8/eval_logs/<name>.log
    python3 dev_fp8/fp8_ssh.py status <name> [n]          # RUNNING/DONE + 日志尾部 n 行(默认 30)
    python3 dev_fp8/fp8_ssh.py kill <name>

密码不落盘到本仓库:取环境变量 FP8_SERVER_PASS,缺省读 ~/.fp8_server_pass(权限 600)。
"""

import base64
import os
import shlex
import subprocess
import sys

HOST = "39.145.28.59"
PORT = "32222"
USER = "root+vm-nIRfDbX7jgamQdwJ"
PASS_FILE = os.path.expanduser("~/.fp8_server_pass")
LOG_DIR = "/root/fp8/eval_logs"

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=20",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=10",
]


def get_password() -> str:
    pw = os.environ.get("FP8_SERVER_PASS")
    if pw:
        return pw
    try:
        with open(PASS_FILE) as f:
            return f.read().strip()
    except OSError:
        sys.exit(f"未找到密码:请设置 FP8_SERVER_PASS 或写入 {PASS_FILE}")


def run_remote(cmd: str, capture: bool = False) -> int:
    env = dict(os.environ, SSHPASS=get_password())
    argv = ["sshpass", "-e", "ssh", *SSH_OPTS, "-p", PORT, f"{USER}@{HOST}", cmd]
    proc = subprocess.run(argv, env=env, capture_output=capture, text=True)
    if capture:
        return proc
    return proc.returncode


def cmd_run(args) -> int:
    return run_remote(" ".join(args))


def cmd_push(args) -> int:
    if len(args) != 2:
        sys.exit("push <local> <remote>")
    env = dict(os.environ, SSHPASS=get_password())
    argv = ["sshpass", "-e", "scp", *SSH_OPTS, "-P", PORT, "-r", args[0], f"{USER}@{HOST}:{args[1]}"]
    return subprocess.run(argv, env=env).returncode


def cmd_pull(args) -> int:
    if len(args) != 2:
        sys.exit("pull <remote> <local>")
    env = dict(os.environ, SSHPASS=get_password())
    argv = ["sshpass", "-e", "scp", *SSH_OPTS, "-P", PORT, "-r", f"{USER}@{HOST}:{args[0]}", args[1]]
    return subprocess.run(argv, env=env).returncode


def cmd_bg(args) -> int:
    if len(args) < 2:
        sys.exit("bg <name> '<shell 命令>'")
    name, payload = args[0], " ".join(args[1:])
    if not name.replace("_", "").replace("-", "").isalnum():
        sys.exit("name 只允许字母数字下划线")
    b64 = base64.b64encode(payload.encode()).decode()
    # base64 传参绕开多层引号;远端落脚本后用 tmux 跑
    remote = (
        f"mkdir -p {LOG_DIR} && "
        f"echo {b64} | base64 -d > /tmp/fp8job_{name}.sh && "
        f"tmux kill-session -t {shlex.quote(name)} 2>/dev/null; "
        f"tmux new-session -d -s {shlex.quote(name)} "
        f"'bash /tmp/fp8job_{name}.sh 2>&1 | tee {LOG_DIR}/{name}.log' && "
        f"echo BG_STARTED {name}"
    )
    return run_remote(remote)


def cmd_status(args) -> int:
    if not args:
        sys.exit("status <name> [n_lines]")
    name = args[0]
    n = args[1] if len(args) > 1 else "30"
    remote = (
        f"tmux has-session -t {shlex.quote(name)} 2>/dev/null && echo STATUS=RUNNING || echo STATUS=DONE; "
        f"echo ---; tail -n {shlex.quote(n)} {LOG_DIR}/{name}.log 2>/dev/null || echo '(无日志)'"
    )
    return run_remote(remote)


def cmd_kill(args) -> int:
    if len(args) != 1:
        sys.exit("kill <name>")
    return run_remote(f"tmux kill-session -t {shlex.quote(args[0])} && echo KILLED")


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    op, args = sys.argv[1], sys.argv[2:]
    table = {
        "run": cmd_run,
        "push": cmd_push,
        "pull": cmd_pull,
        "bg": cmd_bg,
        "status": cmd_status,
        "kill": cmd_kill,
    }
    if op not in table:
        sys.exit(__doc__)
    return table[op](args)


if __name__ == "__main__":
    sys.exit(main())
