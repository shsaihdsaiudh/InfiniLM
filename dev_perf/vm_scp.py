#!/usr/bin/env python3
"""通过堡垒机 scp 推拉文件到/自 5090 租用机。

用法:
    VM_PASSWORD=xxx uv run --no-project --with pexpect python dev_perf/vm_scp.py local1 [local2 ...] remote_dir
    VM_PASSWORD=xxx uv run --no-project --with pexpect python dev_perf/vm_scp.py --pull remote1 [remote2 ...] local_dir

密码从 VM_PASSWORD 环境变量读取（勿把明文密码写进本文件）。
"""
import os
import re
import sys

import pexpect

HOST = "39.145.28.59"
PORT = "32222"
USER = "root+vm-nIRfDbX7jgamQdwJ"
PASSWORD = os.environ.get("VM_PASSWORD", "")


def main() -> int:
    args = sys.argv[1:]
    pull = args and args[0] == "--pull"
    if pull:
        args = args[1:]
    if not PASSWORD:
        print("[!] 需要 VM_PASSWORD 环境变量", file=sys.stderr)
        return 1
    if len(args) < 2:
        print(__doc__, file=sys.stderr)
        return 1
    *srcs, dst = args
    if pull:
        files = " ".join(f'"{USER}@{HOST}:{s}"' for s in srcs)
        target = f'"{dst}"'
    else:
        files = " ".join(f'"{s}"' for s in srcs)
        target = f"{USER}@{HOST}:{dst}"
    scp = (
        f"scp -P {PORT} -o StrictHostKeyChecking=no "
        f"-o UserKnownHostsFile=/dev/null {files} {target}"
    )
    child = pexpect.spawn("/bin/bash", ["-c", scp], encoding="utf-8", timeout=None)
    child.logfile_read = sys.stdout
    re_password = re.compile(r"(?i)password[:：]?\s*$")
    sent = 0
    while True:
        idx = child.expect([re_password, pexpect.EOF], timeout=300)
        if idx == 0:
            sent += 1
            if sent > 2:
                print("\n[!] 密码被拒", file=sys.stderr)
                return 2
            child.sendline(PASSWORD)
        else:
            break
    child.close()
    print(f"\n[vm_scp] exit status: {child.exitstatus}", file=sys.stderr)
    return child.exitstatus or 0


if __name__ == "__main__":
    sys.exit(main())
