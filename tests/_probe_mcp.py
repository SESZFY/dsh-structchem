# -*- coding: utf-8 -*-
"""诊断 MCP stdio 握手：单条 initialize，打印原始回包。"""
import json
import os
import subprocess
import sys

SRC = r"D:\大肥鱼的实验室\dsh-structchem\src"
ENV = dict(os.environ, PYTHONPATH=SRC, PYTHONIOENCODING="utf-8")

p = subprocess.Popen(
    [sys.executable, "-u", "-m", "structchem_mcp.server"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    env=ENV, text=True, encoding="utf-8",
)
msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
       "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                  "clientInfo": {"name": "probe", "version": "0"}}}
p.stdin.write(json.dumps(msg) + "\n")
p.stdin.flush()
print("sent initialize, waiting 20s...", flush=True)
try:
    out, err = p.communicate(input="", timeout=20)
    print("--- stdout ---")
    print(out[:2000] or "(empty)")
    print("--- stderr ---")
    print(err[:3000] or "(empty)")
except subprocess.TimeoutExpired:
    p.kill()
    out, err = p.communicate()
    print("TIMEOUT. stdout:", (out or "")[:800])
    print("stderr:", (err or "")[:3000])
