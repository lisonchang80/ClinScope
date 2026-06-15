# -*- coding: utf-8 -*-
"""ClinScope PreToolUse secret / memory push-guard.

掛在 PreToolUse(Bash)。只在 `git commit` 時動作:掃 staged diff 的內容與檔名,
命中疑似秘密(HF token / kaggle.json key / 私鑰 / API secret)或 user-level memory
路徑就 exit 2 擋下;其餘指令一律放行(exit 0)。

設計原則:**fail-open** —— guard 自身任何錯誤(git 不在、JSON 壞、逾時…)都 exit 0,
絕不因為這支防護腳本的 bug 卡住正常工作流。只有「確實偵測到秘密」才會 exit 2。
"""
import sys, json, re, subprocess
from pathlib import Path

# 強制 utf-8 輸出:hook 可能在 cp950 console 跑,中文擋下訊息若觸發 UnicodeEncodeError
# 會被下面的 fail-open 吞掉 -> 真實秘密反而放行。errors="replace" 保證 stderr 永不崩。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[2]  # .claude/hooks/ -> repo 根

# 疑似秘密的「內容」樣式(掃 staged diff 文字)
SECRET_PATTERNS = [
    (re.compile(r'\bhf_[A-Za-z0-9]{30,}\b'), "Hugging Face token (hf_...)"),
    (re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'), "private key block"),
    (re.compile(r'"key"\s*:\s*"[0-9a-f]{20,}"'), "kaggle.json 形式的 key 欄位"),
    (re.compile(r'(?i)(aws_secret_access_key|api[_-]?secret|client[_-]?secret)\s*[:=]\s*["\']?[A-Za-z0-9/+]{16,}'),
     "API / secret credential"),
]
# 疑似秘密 / 不該入庫的「檔名」(掃 staged 檔名)
BAD_PATH = re.compile(r'(?i)(^|/)(kaggle\.json|\.kaggle/|id_rsa|\.pem$|\.env$)')
# user-level memory 不該進任何 repo
MEM_PATH = re.compile(r'(?i)((^|/)memory/.*\.md$)|((^|/)MEMORY\.md$)')


def main():
    raw = sys.stdin.read()
    i = raw.find("{")                 # 跳過任何前置 BOM/雜訊;hook 輸入 JSON 必由 { 起,免依賴脆弱的隱形字元比對
    raw = raw[i:] if i >= 0 else raw
    data = json.loads(raw) if raw.strip() else {}
    cmd = (data.get("tool_input") or {}).get("command", "") or ""

    # 只攔 git commit(真正決定「什麼進版控」的點);其餘秒放行
    if not re.search(r'\bgit\b[^|&;]*\bcommit\b', cmd):
        sys.exit(0)

    # encoding="utf-8" 是關鍵:預設 text=True 用 locale(cp950)解碼 git 輸出,
    # 而 ClinScope 的 diff 幾乎都含 utf-8 中文 -> cp950 解碼炸 UnicodeDecodeError
    # -> 被 fail-open 吞掉 -> guard 形同虛設。errors="replace" 確保永不因解碼崩潰。
    gkw = dict(cwd=REPO, capture_output=True, encoding="utf-8", errors="replace")
    names = subprocess.run(["git", "diff", "--cached", "--name-only"], timeout=10, **gkw).stdout.splitlines()
    diff = subprocess.run(["git", "diff", "--cached", "--unified=0"], timeout=20, **gkw).stdout

    hits = []
    for n in names:
        if BAD_PATH.search(n):
            hits.append(f"  - 檔名疑似秘密: {n}")
        if MEM_PATH.search(n):
            hits.append(f"  - user-level memory 不該入庫: {n}")
    for pat, desc in SECRET_PATTERNS:
        if pat.search(diff):
            hits.append(f"  - diff 內容命中: {desc}")

    if hits:
        sys.stderr.write(
            "[ClinScope secret-guard] 擋下這次 git commit —— staged 變更疑似含秘密 / 不該入庫的內容:\n"
            + "\n".join(dict.fromkeys(hits))  # 去重保序
            + "\n\n請先 `git restore --staged <file>` 移除,或確認誤判後改用其他方式提交。\n"
            "(此 guard 來自 .claude/hooks/guard_secrets.py;確定誤判可移除該 hook。)\n")
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # fail-open:guard 自身任何錯誤都不擋使用者
        sys.exit(0)
