"""Check all dashboard JavaScript with Node, without loading a live camera server."""
from pathlib import Path
import re
import subprocess

root = Path(__file__).parent
html = (root / "index.html").read_text(encoding="utf-8")
scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", html, re.S)
subprocess.run(["node", "--check"], input="\n".join(scripts), encoding="utf-8", check=True)
subprocess.run(["node", "--check", str(root / "workspace.js")], check=True)
print("Dashboard JavaScript syntax checks passed")
