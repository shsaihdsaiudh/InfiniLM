#!/usr/bin/env python3
"""通过堡垒机 SSH 登录 5090 租用机并执行命令（pexpect 状态机）。

用法:
    VM_PASSWORD=xxx uv run --no-project --with pexpect python dev_perf/vm_ssh.py "hostname && nvidia-smi -L"
    VM_OTP=<验证码> ...   # 若堡垒机出现二次验证提示

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

CMD_MARKER_BEGIN = "__VM_CMD_BEGIN__"
CMD_MARKER_END = "__VM_CMD_END__"


def main() -> int:
    if not PASSWORD:
        print("[!] 需要 VM_PASSWORD 环境变量", file=sys.stderr)
        return 1
    remote_cmd = sys.argv[1] if len(sys.argv) > 1 else "hostname"
    ssh = (
        f"ssh -p {PORT} -o StrictHostKeyChecking=no "
        f"-o UserKnownHostsFile=/dev/null -o PubkeyAuthentication=yes {USER}@{HOST}"
    )
    child = pexpect.spawn(ssh, encoding="utf-8", timeout=60, maxread=65536)
    child.logfile_read = sys.stdout

    # 状态机：依次响应可能出现的提示；任何未知输出都会被 logfile_read 原样打印
    re_password = re.compile(r"(?i)password[:：]?\s*$|密码[:：]?\s*$")
    re_verify = re.compile(r"(?i)verification code|otp|token|验证码|动态口令|二次验证")
    re_yesno = re.compile(r"\(yes/no(/\[fingerprint]\))?\?\s*$", re.S)
    re_menu = re.compile(r"(?i)select|choose|输入序号|请选择")
    # 常见 shell 提示符：root@...:# 、 [root@...]# 、 ~# 、bash-5.1# 等
    re_prompt = re.compile(r"[#$]\s*$")

    password_sent = 0
    logged_in = False
    idle_deadline = 300  # 整体最长等待（秒）

    while True:
        idx = child.expect(
            [re_password, re_verify, re_yesno, re_menu, re_prompt,
             pexpect.EOF, pexpect.TIMEOUT],
            timeout=60,
        )
        if idx == 0:  # password
            password_sent += 1
            if password_sent > 3:
                print("\n[!] 密码提示超过 3 次，可能密码错误", file=sys.stderr)
                return 2
            child.sendline(PASSWORD)
        elif idx == 1:  # 二次验证（验证码/OTP）
            code = os.environ.get("VM_OTP", "")
            print(f"\n[!] 出现二次验证提示，VM_OTP={'已提供' if code else '未提供'}", file=sys.stderr)
            if not code:
                print("[!] 请用 VM_OTP=<验证码> 重跑", file=sys.stderr)
                return 3
            child.sendline(code)
        elif idx == 2:  # host key 确认
            child.sendline("yes")
        elif idx == 3:  # 堡垒机菜单：默认选 1
            print("\n[!] 检测到菜单，默认选择 1", file=sys.stderr)
            child.sendline("1")
        elif idx == 4:  # shell 提示符
            logged_in = True
            break
        elif idx == 5:  # EOF
            print("\n[!] 连接在登录前关闭", file=sys.stderr)
            return 4
        else:  # TIMEOUT
            idle_deadline -= 60
            if idle_deadline <= 0:
                print("\n[!] 等待登录超时", file=sys.stderr)
                return 5

    # 已登录，执行命令（包 marker 便于定位输出）
    child.sendline(f'echo {CMD_MARKER_BEGIN}; {remote_cmd}; echo {CMD_MARKER_END}:$?')
    output_lines = []
    while True:
        idx = child.expect([re.compile(re.escape(CMD_MARKER_END) + r":(\d+)"),
                            pexpect.EOF, pexpect.TIMEOUT], timeout=120)
        output_lines.append(child.before)
        if idx == 0:
            rc = int(child.match.group(1))
            break
        if idx == 1:
            rc = -1
            break
        # TIMEOUT：长命令继续等
    child.sendline("exit")
    child.close()
    print(f"\n[vm_ssh] remote exit code: {rc}", file=sys.stderr)
    return 0 if logged_in else 6


if __name__ == "__main__":
    sys.exit(main())
