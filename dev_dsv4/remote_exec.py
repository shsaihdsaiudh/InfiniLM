#!/usr/bin/env python3
"""SSH helper for running commands on the DeepSeek-V4 dev machine.

Wraps the system OpenSSH client (the rental platform's SSH proxy drops
paramiko sessions, so we drive ``ssh``/``scp`` with SSH_ASKPASS instead).
All connection parameters come from environment variables so no host,
username, or password is ever stored in the repository:

    DSV4_SSH_HOST   (required)  server hostname or IP
    DSV4_SSH_PORT   (optional, default 22)
    DSV4_SSH_USER   (optional, default root)
    DSV4_SSH_PASS   (required)  password

Usage:
    python dev_dsv4/remote_exec.py exec "nvidia-smi"
    python dev_dsv4/remote_exec.py put local/path/test.py /remote/path/test.py
    python dev_dsv4/remote_exec.py get /remote/path/log.txt local/log.txt

``exec`` streams stdout/stderr and exits with the remote command's exit code.
"""

import os
import stat
import subprocess
import sys
import tempfile

ASKPASS_SCRIPT = "#!/bin/sh\nprintf '%s' \"$DSV4_SSH_PASS\"\n"

COMMON_OPTIONS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=15",
    "-o", "PreferredAuthentications=password",
    "-o", "PubkeyAuthentication=no",
    "-o", "NumberOfPasswordPrompts=1",
]


def connection_env():
    host = os.environ.get("DSV4_SSH_HOST")
    password = os.environ.get("DSV4_SSH_PASS")
    if not host or not password:
        sys.stderr.write("DSV4_SSH_HOST and DSV4_SSH_PASS must be set\n")
        sys.exit(2)
    askpass = tempfile.NamedTemporaryFile(
        mode="w", prefix="dsv4-askpass-", delete=False
    )
    askpass.write(ASKPASS_SCRIPT)
    askpass.close()
    os.chmod(askpass.name, stat.S_IRWXU)
    env = dict(os.environ)
    env["SSH_ASKPASS"] = askpass.name
    env["SSH_ASKPASS_REQUIRE"] = "force"
    return env, askpass.name


def target():
    host = os.environ["DSV4_SSH_HOST"]
    user = os.environ.get("DSV4_SSH_USER", "root")
    return f"{user}@{host}"


def port():
    return os.environ.get("DSV4_SSH_PORT", "22")


def main():
    if len(sys.argv) < 3:
        sys.stderr.write(__doc__)
        sys.exit(2)
    mode = sys.argv[1]
    env, askpass_path = connection_env()
    try:
        if mode == "exec":
            cmd = ["ssh", *COMMON_OPTIONS, "-p", port(), target(), sys.argv[2]]
        elif mode == "put" and len(sys.argv) == 4:
            cmd = [
                "scp",
                *COMMON_OPTIONS,
                "-P",
                port(),
                sys.argv[2],
                f"{target()}:{sys.argv[3]}",
            ]
        elif mode == "get" and len(sys.argv) == 4:
            cmd = [
                "scp",
                *COMMON_OPTIONS,
                "-P",
                port(),
                f"{target()}:{sys.argv[2]}",
                sys.argv[3],
            ]
        else:
            sys.stderr.write(__doc__)
            sys.exit(2)
        process = subprocess.run(cmd, env=env, check=False)
        sys.exit(process.returncode)
    finally:
        os.unlink(askpass_path)


if __name__ == "__main__":
    main()
