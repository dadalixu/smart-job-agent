# -*- coding: utf-8 -*-
"""带记忆的求职助手 Agent

不依赖 LangChain 等框架，手写「意图理解 → 工具调用 → 结果回灌 → 终止判断」闭环，
带三类持久化记忆：用户画像 / 岗位库 / 搜索词。

命令：
  python agent.py                     交互式对话
  python agent.py selftest            离线自测（不联网、不需密钥；改完代码先跑它）
  python agent.py diag                联网自检（一键区分代码问题 / 网络问题）
  python agent.py demo                内置演示
  python agent.py clean [--dry-run]   清洗脏数据（自动带时间戳备份）
  python agent.py restore [备份名]    从备份恢复（默认取最近一份）
  python agent.py backups             列出所有备份

配置：密钥通过环境变量 DEEPSEEK_API_KEY / TAVILY_API_KEY 提供，见 .env.example。
数据：全部记忆写入同目录下的 job_prefs.json，写前自动滚动备份到 backups/。
"""

import json
import logging
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
from datetime import datetime
from urllib.parse import urlparse

from openai import OpenAI
from tavily import TavilyClient

# ---------- 基础配置 ----------
MODEL = "deepseek-chat"
MAX_STEPS = 10               # 单轮最多几次工具调用，防死循环
USE_LLM_EXTRACT = True       # 关掉则退回「原始网页标题」模式
MAX_JOBS = 500               # 岗位记忆上限，超出按 found_at 淘汰最旧
TOOL_RESULT_LIMIT = 8000     # 单个工具结果回灌模型的最大字符数
MAX_RESULTS = 10             # 单次联网搜索最多取几条
MAX_HISTORY_TURNS = 20       # 多轮对话只保留最近 N 轮
SCHEMA_VERSION = 2
BACKUP_DIRNAME = "backups"
BACKUP_KEEP = 5              # 备份只保留最近 N 份

# 三个超时的关系（配错会既慢又误报）：
#   1) search_depth="advanced" 本身要 20~60s，API 超时设小会在网络正常时误报超时；
#   2) tavily-python 会按 IP 逐个重试，实际耗时 ≈ IP数 × timeout，成倍放大；
#   3) 所以先用 3 秒 TCP 预检判断「能不能连上」，连得上再给足 60 秒。
TAVILY_TIMEOUT = 60
TAVILY_MAX_ATTEMPTS = 1
TAVILY_BASE_URL = os.environ.get("TAVILY_BASE_URL", "https://api.tavily.com")
PRECHECK_TIMEOUT = 3
DIAG_TIMEOUT = 8
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEM_FILE = os.path.join(BASE_DIR, "job_prefs.json")          # 记忆数据（唯一写入目标）
BACKUP_DIR = os.path.join(BASE_DIR, BACKUP_DIRNAME)


def _load_dotenv():
    """轻量加载同目录 .env（不引入 python-dotenv）；已有的系统环境变量优先。"""
    path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip("\"'"))
    except OSError as e:
        logger.warning("读取 .env 失败：%s", e)


_load_dotenv()

_LOCK = threading.RLock()
_client = None


def get_client():
    """懒加载 DeepSeek 客户端（模块顶层直接取环境变量会在 import 期就崩）。"""
    global _client
    if _client is None:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY。\n"
                               '  PowerShell :  $env:DEEPSEEK_API_KEY="sk-xxx"\n'
                               "  CMD        :  set DEEPSEEK_API_KEY=sk-xxx")
        _client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
    return _client


class SearchUnavailable(RuntimeError):
    """搜索服务不可用（连不上 / 超时 / 配额），用于和「搜到了但没结果」区分开。"""


# ---------- 记忆存储：安全读 + 原子写 + 回退链 ----------
def _default_memory():
    return {"schema_version": SCHEMA_VERSION, "profile": {}, "jobs": [], "queries": []}


def _read_json(path):
    """安全读取 JSON 对象；空文件 / 损坏 / 顶层非对象一律返回 None。

    调用方据此区分「没有数据」和「数据不可用」。
    旧版遇到 0 字节文件时 json.load 报错后静默返回空记忆 ——
    数据被清空了，程序却装作一切正常，所以这层判断是必须的。
    """
    if not path or not os.path.exists(path):
        return None
    try:
        if os.path.getsize(path) == 0:
            logger.warning("%s 是 0 字节（可能被外部清空），按不可用处理",
                           os.path.basename(path))
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        logger.warning("读取 %s 失败：%s", os.path.basename(path), e)
        return None
    if not isinstance(data, dict):
        logger.warning("%s 顶层不是对象，忽略", os.path.basename(path))
        return None
    return data


def _memory_sources():
    """回退链：主文件 → 主文件 .bak。"""
    return [MEM_FILE, MEM_FILE + ".bak"]


def _normalize_memory(data):
    """补齐缺失字段，保证后续代码可以放心下标访问。"""
    if not isinstance(data.get("profile"), dict):
        data["profile"] = {}
    for key in ("jobs", "queries"):
        if not isinstance(data.get(key), list):
            data[key] = []
    data.setdefault("schema_version", SCHEMA_VERSION)
    return data


def load_memory():
    """读取全部记忆。内部函数，不要注册成模型工具。

    主文件为空或损坏时不会假装「没有记忆」，而是沿回退链找回数据并告警。
    """
    with _LOCK:
        for path in _memory_sources():
            data = _read_json(path)
            if data is None:
                continue
            if path != MEM_FILE:
                logger.warning("主记忆文件不可用，已回退读取 %s（%d 条岗位）；"
                               "下次写入会自动修复主文件",
                               os.path.basename(path), len(data.get("jobs") or []))
            return _normalize_memory(data)
        return _default_memory()


def _prune_backups():
    """只保留最近 BACKUP_KEEP 份备份。"""
    try:
        files = sorted(f for f in os.listdir(BACKUP_DIR)
                       if f.startswith("job_prefs-") and f.endswith(".json"))
    except OSError:
        return
    for name in files[:-BACKUP_KEEP]:
        try:
            os.remove(os.path.join(BACKUP_DIR, name))
        except OSError:
            pass


def _backup_mem_file(tag=""):
    """把当前记忆滚动备份到 backups/，返回备份路径（同一秒内只写一份）。"""
    if not os.path.exists(MEM_FILE) or os.path.getsize(MEM_FILE) == 0:
        return None
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        suffix = f"-{tag}" if tag else ""
        dst = os.path.join(BACKUP_DIR,
                           f"job_prefs-{datetime.now():%Y%m%d-%H%M%S}{suffix}.json")
        if not os.path.exists(dst):
            shutil.copy(MEM_FILE, dst)
        _prune_backups()
        return dst
    except OSError as e:
        logger.warning("备份失败（不影响写入）：%s", e)
        return None


def _atomic_write(data, backup=True):
    """写盘三道保险：

      1) 先 json.dumps 到内存 —— 序列化失败就不会碰到主文件；
      2) 写完校验临时文件非空 —— 空内容拒绝替换；
      3) 替换前滚动备份一份 —— 万一被外部清空还能救回来。
    backup=False 用于调用方自己已经备过份的场景（如 clean）。
    """
    with _LOCK:
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        if not payload.strip() or payload.strip() in ("{}", "null", "[]"):
            raise ValueError("拒绝写入空内容（上游数据异常，已放弃落盘）")
        if backup:
            _backup_mem_file()

        fd, tmp = tempfile.mkstemp(dir=BASE_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            if os.path.getsize(tmp) == 0:
                raise OSError("临时文件写入为空，已放弃替换主文件")
            os.replace(tmp, MEM_FILE)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            raise


def list_backups():
    """列出 backups/ 里的备份（新 → 旧）。"""
    if not os.path.isdir(BACKUP_DIR):
        return []
    return sorted((os.path.join(BACKUP_DIR, f) for f in os.listdir(BACKUP_DIR)
                   if f.startswith("job_prefs-") and f.endswith(".json")),
                  reverse=True)


def restore(path=None):
    """从备份恢复记忆文件；不指定则取最近一份。恢复前先把当前文件另存一份。"""
    backups = list_backups()
    if not backups:
        print(f"没有可用备份（{BACKUP_DIRNAME}/ 为空）。")
        return False
    src = path or backups[0]
    if _read_json(src) is None:
        print(f"备份不存在或已损坏：{src}")
        return False
    if os.path.exists(MEM_FILE) and os.path.getsize(MEM_FILE) > 0:
        _backup_mem_file(tag="before-restore")
    shutil.copy(src, MEM_FILE)
    mem = load_memory()
    print(f"已从 {os.path.relpath(src, BASE_DIR)} 恢复："
          f"{len(mem.get('jobs') or [])} 条岗位、{len(mem.get('queries') or [])} 条搜索词")
    return True


# ---------- 用户画像 ----------
_SPLIT_RE = re.compile(r"[、,，;；/]")


def save_memory(key, value, mode="overwrite"):
    """保存画像字段。mode=append 时按「、,，;；/」拆分多值并去重。"""
    if mode not in ("overwrite", "append"):
        mode = "overwrite"
    with _LOCK:
        mem = load_memory()
        profile = mem["profile"]
        if mode == "append":
            existing = profile.get(key, [])
            if not isinstance(existing, list):
                existing = [existing]
            incoming = value if isinstance(value, list) else _SPLIT_RE.split(str(value))
            for v in (x.strip() for x in incoming):
                if v and v not in existing:
                    existing.append(v)
            profile[key] = existing
        else:
            profile[key] = value
        _atomic_write(mem)
    return f"已保存 {key}={value} (mode={mode})"


def get_profile():
    """模型可见的画像读取：只返回 profile，不把岗位记录一起塞进上下文。"""
    return json.dumps(load_memory().get("profile", {}), ensure_ascii=False)


# ---------- 归一化与去重键 ----------
# 站点品牌名，归一化和括号处理共用
_SITE_BRANDS = (
    "boss直聘|boss|智联招聘|智联|前程无忧|51job|猎聘|拉勾|"
    "脉脉|看准网|看准|职友集|实习僧|牛客|应届生求职网|应届生|大街|领英|linkedin|"
    "招聘信息网|招聘网|就业信息网|求职网|人才网|招聘详情|招聘列表|"
    "watchjobs|猎头合作交易平台|招聘信息"
)
# 标题尾部形如「- BOSS直聘」「_ 应届生求职网」的站点名，整段删掉
_NOISE_TAILS = re.compile(rf"[-|_—–｜]\s*(?:{_SITE_BRANDS}).*$", re.IGNORECASE)
# 括号分两类处理：
#   ① 纯噪声括号 —— 里面只有「职位信息」「急招」或站点名，整体删掉；
#   ② 其余括号 —— 只去掉括号符号、保留内容，因为里面往往是岗位名或公司名。
# 旧版一刀切删掉所有括号内容，遇到「【岗位名】-站点名」会把岗位名一起删没，
# 去重键直接退化 —— 这是自测当场抓出来的真实缺陷。
_BRACKET_NOISE = re.compile(
    rf"[（(\[【]\s*(?:职位信息|招聘信息|招聘职位|招聘岗位|岗位信息|职位描述|"
    rf"急招|热招|最新|置顶|推荐|精选|热门|hot|new|{_SITE_BRANDS})\s*[)）\]】]",
    re.IGNORECASE)
_BRACKET_ONLY = re.compile(r"[（(\[【)）\]】]")
# 兜底：剥掉结尾的通用词，不影响前面的岗位名
_GENERIC_TAIL = re.compile(
    r"(?:招聘信息网|就业信息网|求职网|人才网|招聘网|招聘信息|招聘详情|招聘列表|"
    r"招聘职位|招聘岗位|岗位信息|职位信息|招聘)+$", re.IGNORECASE)
_PUNCT = re.compile(r"[\-—_–｜|·•,，。.:：/\\]+")
_KEEP = re.compile(r"[^\w\u4e00-\u9fff]")


def _normalize(text):
    """归一化标题/公司/地点。顺序很关键：
    先删噪声括号 → 其余括号去符号 → 剥尾部站点名 → 去空白 → 剥结尾通用词 → 去标点。
    """
    t = (text or "").lower()
    t = _BRACKET_NOISE.sub("", t)
    t = _BRACKET_ONLY.sub("", t)
    t = _NOISE_TAILS.sub("", t)
    t = re.sub(r"[\s\u3000]+", "", t)
    t = _GENERIC_TAIL.sub("", t)
    t = _PUNCT.sub("", t)
    return _KEEP.sub("", t).strip()


def _url_key(url):
    """URL 归一化：小写、去 www.、去 query/fragment、去尾斜杠。"""
    try:
        p = urlparse(url or "")
    except Exception:
        return ""
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = (p.path or "").rstrip("/").lower()
    return f"{host}{path}" if (host or path) else ""


def _job_key(job):
    """去重键。键在每次读取时重算，所以改算法不需要数据迁移。

      1) 有 URL    → ("U", url_key)：同一个 URL 一定是同一条岗位
      2) 无 URL    → ("C", 公司, 岗位, 地点)：内容比对
      3) 只有标题  → ("T", 标题)
    """
    url_key = _url_key(job.get("source") or "")
    if url_key:
        return ("U", url_key)
    company = _normalize(job.get("company") or "")
    title = _normalize(job.get("title") or "")
    location = _normalize(job.get("location") or "")
    if company and title:
        return ("C", company, title, location)
    return ("T", title) if title else None


def _index_jobs(jobs):
    """建立「去重键 → 岗位记录」索引，供查重与字段补全使用。"""
    index = {}
    for j in jobs:
        k = _job_key(j)
        if k and k not in index:
            index[k] = j
    return index


def _merge_job(old, new):
    """同一条岗位再次出现时，用新记录补全旧记录缺失的字段（只补空值，不覆盖已有）。"""
    changed = False
    for field in ("company", "location"):
        if not str(old.get(field) or "").strip() and str(new.get(field) or "").strip():
            old[field] = str(new[field]).strip()
            changed = True
    return changed


def _trim_jobs(jobs):
    """岗位数量上限，超出按 found_at 淘汰最旧。"""
    if len(jobs) <= MAX_JOBS:
        return jobs
    return sorted(jobs, key=lambda x: x.get("found_at") or "")[-MAX_JOBS:]


def _save_jobs(jobs, query=None):
    """批量写入岗位，返回 (新增列表, 已存在列表)。"""
    if not jobs and not query:
        return [], []
    with _LOCK:
        mem = load_memory()
        index = _index_jobs(mem["jobs"])
        now = datetime.now().isoformat(timespec="seconds")
        added, dup = [], []
        dirty = False                # 显式脏标记，不靠「回头看内存」判断

        for j in jobs or []:
            if not isinstance(j, dict):
                continue
            j = dict(j)
            for field in ("company", "location", "source", "snippet"):
                j.setdefault(field, "")
            j["query"] = j.get("query") or query
            j["found_at"] = j.get("found_at") or now

            key = _job_key(j)
            if not key:
                continue

            old = index.get(key)
            if old is not None:
                dup.append(j)
                if _merge_job(old, j):        # 新记录能补全旧记录缺的字段
                    dirty = True
                    index.pop(key, None)      # 字段变了，键可能跟着变，需重新入索引
                    index[_job_key(old)] = old
                continue

            mem["jobs"].append(j)
            index[key] = j
            added.append(j)
            dirty = True

        if query and query not in mem["queries"]:
            mem["queries"].append(query)
            dirty = True

        if dirty:
            mem["jobs"] = _trim_jobs(mem["jobs"])
            _atomic_write(mem)
    return added, dup


def save_job(title, company="", location="", source="", query=None):
    if not title:
        return "保存失败：岗位名称不能为空。"
    added, dup = _save_jobs([{"title": title, "company": company,
                              "location": location, "source": source}], query=query)
    if added:
        return f"已保存岗位：{title}"
    return f"该岗位已存在，未重复保存：{title}" if dup else "保存失败：岗位信息无效。"


def list_jobs(limit=20):
    jobs = load_memory()["jobs"]
    if not jobs:
        return "记忆里还没有岗位记录。"
    try:
        limit = max(1, min(int(limit) if limit else 20, 200))    # 防止负数切片
    except (TypeError, ValueError):
        limit = 20
    shown = jobs[-limit:]
    return (f"共 {len(jobs)} 条，显示最近 {len(shown)} 条：\n"
            + json.dumps(shown, ensure_ascii=False, indent=2))


# ---------- 官网过滤 ----------
# 只在 host 上匹配的纯域名（中文品牌名放这里永远不会命中，所以分开存）
JOB_AGGREGATOR_DOMAINS = {
    "zhipin.com", "zhaopin.com", "51job.com", "liepin.com", "lagou.com",
    "maimai.cn", "58.com", "ganji.com", "dajie.com", "linkedin.com",
    "shixiseng.com", "nowcoder.com", "yingjiesheng.com", "kanzhun.com",
    "jobui.com", "zhihu.com", "xiaohongshu.com", "csdn.net", "juejin.cn",
}
# 中文品牌名在标题/正文里匹配
JOB_AGGREGATOR_BRANDS = (
    "boss直聘", "智联招聘", "前程无忧", "猎聘", "拉勾", "脉脉",
    "看准网", "职友集", "实习僧", "牛客", "应届生求职网",
)
CAREER_URL_PATTERNS = [r"/careers?", r"/jobs?", r"/join", r"/recruit",
                       r"/zhaopin", r"/hr\b", r"/talent", r"/campus"]
CAREER_HOST_PREFIXES = ["careers.", "jobs.", "job.", "hr.",
                        "zhaopin.", "recruit.", "talent."]


def _host(url):
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _is_aggregator(url):
    """域名精确 / 后缀匹配：58.com 命中，158.com 不命中。"""
    host = _host(url)
    return bool(host) and any(host == d or host.endswith("." + d)
                              for d in JOB_AGGREGATOR_DOMAINS)


def _has_brand_noise(text):
    return any(b in (text or "") for b in JOB_AGGREGATOR_BRANDS)


def _looks_like_career_page(url):
    """careers./jobs. 子域，或路径含 /careers、/jobs、/join 的页面。"""
    if not url:
        return False
    host, path = _host(url), urlparse(url).path.lower()
    return (any(host.startswith(p) for p in CAREER_HOST_PREFIXES)
            or any(re.search(p, path) for p in CAREER_URL_PATTERNS))


def _filter_official(results):
    """过滤聚合站，返回 (疑似官网, 其他)。"""
    official, others = [], []
    for r in results:
        url = r.get("url") or ""
        if not url or _is_aggregator(url) or _has_brand_noise(r.get("title")):
            continue
        (official if _looks_like_career_page(url) else others).append(r)
    return official, others


# ---------- 联网搜索 ----------
# 排除聚合站（旧版写过 sorted(...)[:8]，等于按字母序随便砍掉一半，这里是全量排除）
_SITE_EXCLUDES = " ".join(f"-site:{d}" for d in sorted(JOB_AGGREGATOR_DOMAINS))

_PROXY_ENV_KEYS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                   "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy")


def _active_proxies():
    """列出当前真正生效的代理环境变量（requests / openai 都会自动读这些）。"""
    return {k: os.environ[k] for k in _PROXY_ENV_KEYS if os.environ.get(k)}


def _net_hint(err):
    """把底层网络异常翻译成能直接照着做的排查提示。"""
    host = urlparse(TAVILY_BASE_URL).hostname or TAVILY_BASE_URL
    active = _active_proxies()
    lines = [f"搜索服务不可用 —— {type(err).__name__}: {str(err)[:180]}",
             f"  目标地址：{host}   超时：{TAVILY_TIMEOUT}s   "
             f"已尝试：{TAVILY_MAX_ATTEMPTS} 次"]
    if active:
        lines.append("  当前生效的代理变量：" +
                     "；".join(f"{k}={v}" for k, v in active.items()))
        lines.append("  → 代理连不上 Tavily 就关掉它改直连，或换一个能用的代理")
    else:
        lines.append("  当前生效的代理变量：无（走直连）")
        lines.append(f"  → 直连不通说明 {host} 在你这条网络下被拦了，需要开代理")
    lines.append("  一键自检：python agent.py diag")
    return "\n".join(lines)


def _tcp_ok(host, port=443, timeout=PRECHECK_TIMEOUT):
    """快速 TCP 预检。不通就立刻返回 False，不去干等 API 超时。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _probe(host, port=443, timeout=DIAG_TIMEOUT):
    """探测单个主机的 DNS 与 TCP 握手，返回 (结果行列表, 是否连通)。"""
    try:
        ip = socket.gethostbyname(host)
    except Exception as e:
        return [("DNS", False, type(e).__name__)], False
    t0 = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return [("DNS", True, ip),
                    (f"TCP:{port}", True, f"{time.time() - t0:.2f}s")], True
    except OSError as e:
        return [("DNS", True, ip),
                (f"TCP:{port}", False,
                 f"{type(e).__name__} {time.time() - t0:.2f}s")], False


def diag():
    """一键自检：把「代码问题」和「网络问题」分开，并给出下一步动作。"""
    print("=" * 62)
    print("agent.py 自检")
    print(f"  Python : {sys.executable}")

    targets = [("DeepSeek（模型）", urlparse(DEEPSEEK_BASE_URL).hostname,
                "DEEPSEEK_API_KEY"),
               ("Tavily（联网搜索）", urlparse(TAVILY_BASE_URL).hostname,
                "TAVILY_API_KEY")]
    reachable = {}
    for label, host, key_name in targets:
        key = os.environ.get(key_name)
        print(f"\n[{label}]  {host}")
        print(f"  {key_name:<16}: "
              + (f"已设置（{key[:10]}…）" if key else "❌ 未设置"))
        rows, ok = _probe(host)
        for name, good, detail in rows:
            print(f"  {name:<16}: {'✅' if good else '❌'} {detail}")
        reachable[label] = ok

    active = _active_proxies()
    print("\n代理变量："
          + ("；".join(f"{k}={v}" for k, v in active.items()) if active else "无"))
    print("\n--- 备选搜索源（Tavily 不通时可考虑）---")
    for label, host in (("博查 Bocha", "api.bochaai.com"),
                        ("必应 cn.bing.com", "cn.bing.com")):
        print(f"  {label:<18}: "
              + ("✅ 可达" if _tcp_ok(host, 443, DIAG_TIMEOUT) else "❌ 不可达"))

    ds, tv = reachable["DeepSeek（模型）"], reachable["Tavily（联网搜索）"]
    print("\n" + "-" * 62)
    if not ds:
        print("结论：连 DeepSeek 都不通，先解决基础网络（换网 / 开代理）再看别的。")
    elif tv:
        print("结论：两个服务都通。若 find_job 仍报错，多半是密钥或配额问题，")
        print("      去 Tavily 控制台确认这个 key 还有额度。")
    else:
        print("结论：模型通、搜索不通 —— 不是代码问题，是 Tavily 服务器连不上。")
        print("      三条路任选一条：")
        print("        A. 开梯子 / VPN，重跑本自检确认 TCP 变成 ✅")
        print("        B. 换国内可直连的搜索后端（博查等），无需代理")
        print("        C. set TAVILY_BASE_URL=https://你的转发地址")
    print("-" * 62)
    return {"deepseek": ds, "tavily": tv}


def _search_web(query, max_results=MAX_RESULTS):
    """调用 Tavily；失败时抛出带排查提示的 SearchUnavailable。"""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 TAVILY_API_KEY，无法联网搜索。")
    host = urlparse(TAVILY_BASE_URL).hostname or TAVILY_BASE_URL

    # 先做 3 秒 TCP 预检：少了这一步，连接被拦时 requests 会逐个重试 IP，
    # 卡住「IP 数 × timeout × 尝试次数」秒（实测能到 90 秒）。
    if not _tcp_ok(host):
        raise SearchUnavailable(_net_hint(ConnectionError(
            f"TCP 预检失败：{host}:443 在 {PRECHECK_TIMEOUT} 秒内没握手成功")))

    last_err = None
    for attempt in range(1, TAVILY_MAX_ATTEMPTS + 1):
        try:
            return TavilyClient(api_key=api_key, base_url=TAVILY_BASE_URL,
                                timeout=TAVILY_TIMEOUT).search(
                query=query, search_depth="advanced", max_results=max_results)
        except Exception as e:
            last_err = e
            logger.warning("搜索第 %d/%d 次尝试失败：%s：%s",
                           attempt, TAVILY_MAX_ATTEMPTS, type(e).__name__, e)

    raise SearchUnavailable(_net_hint(last_err)) from last_err


def _parse_jobs(results, query):
    """兜底路径：直接用原始网页标题，不做公司抽取。"""
    return [{"title": str(r.get("title") or "").strip(), "company": "",
             "location": "", "source": r.get("url") or "",
             "snippet": (r.get("content") or "").strip()[:300], "query": query}
            for r in results if str(r.get("title") or "").strip()]


EXTRACT_SYSTEM = """你是招聘信息结构化助手。
输入是一个 JSON 数组，每项包含 idx、title、url、content。
请输出 JSON 对象：{"items":[{"idx":0,"title":"","company":"","location":""}]}
要求：
- idx 必须与输入一一对应，items 数量与输入相同。
- title：只保留岗位名称，去掉站点名、【】[]、"- XX招聘信息" 这类噪声。
- company：必须是 title 或 content 中真实出现过的公司全称；没有出现就填空字符串。
- location：只填城市名，例如「武汉」「杭州」；不确定填空字符串。
- 禁止编造任何字段。"""


def _extract_fields(results):
    """用模型把标题/公司/地点规范化。失败返回 None，由调用方回退。"""
    if not USE_LLM_EXTRACT or not results:
        return None
    payload = [{"idx": i, "title": r.get("title", ""), "url": r.get("url", ""),
                "content": (r.get("content") or "")[:400]}
               for i, r in enumerate(results)]
    try:
        resp = get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": EXTRACT_SYSTEM},
                      {"role": "user",
                       "content": json.dumps(payload, ensure_ascii=False)}],
            response_format={"type": "json_object"}, temperature=0, timeout=60)
        items = json.loads(resp.choices[0].message.content or "{}").get("items") or []
        if not isinstance(items, list):
            return None
    except Exception:
        logger.exception("结构化抽取失败，回退到原始标题")
        return None

    by_idx = {}
    for it in items:
        if isinstance(it, dict):
            try:
                by_idx[int(it.get("idx", -1))] = it
            except (TypeError, ValueError):
                pass

    out = []
    for i, r in enumerate(results):
        it = by_idx.get(i, {})
        out.append({"title": (it.get("title") or r.get("title") or "").strip(),
                    "company": (it.get("company") or "").strip(),
                    "location": (it.get("location") or "").strip(),
                    "source": r.get("url") or "",
                    "snippet": (r.get("content") or "").strip()[:300]})
    return out


def _format_jobs(jobs, header):
    """渲染岗位列表。全部走 .get()，缺字段也不会 KeyError。"""
    lines = [header]
    for j in jobs:
        bits = [str(b).strip() for b in (j.get("company"), j.get("location"))
                if str(b or "").strip()]
        suffix = f"（{' / '.join(bits)}）" if bits else ""
        lines.append(f"- {str(j.get('title') or '（无标题）').strip()}{suffix}\n"
                     f"  {str(j.get('source') or '').strip() or '（无链接）'}")
    return "\n".join(lines)


def find_job(query):
    """联网搜岗位：过滤聚合站 → 结构化抽取 → 去重入库。"""
    searched_before = query in load_memory().get("queries", [])

    try:
        response = _search_web(f"{query} {_SITE_EXCLUDES}")
    except SearchUnavailable as e:
        return str(e)                       # 已内置排查提示
    except RuntimeError as e:
        return f"错误：{e}"
    except Exception as e:
        logger.exception("搜索失败")
        return f"搜索失败（{type(e).__name__}）：无法获取最新招聘信息。"

    official, others = _filter_official(response.get("results", []) or [])
    picked = official if official else others
    if not picked:
        return ("未找到相关岗位（已过滤招聘聚合站）。\n"
                "建议：直接提供目标公司名，我再搜它的官网招聘页。")

    added, dup = _save_jobs(_extract_fields(picked) or _parse_jobs(picked, query),
                            query=query)
    parts = []
    if searched_before:
        parts.append(f"（提示：搜索词「{query}」此前搜过，本次重新联网核对）")
    parts.append(_format_jobs(added, "【本次新发现】") if added
                 else "【本次新发现】\n无新岗位。")
    if dup:
        parts.append(_format_jobs(dup, "【此前已记录，不重复计入】"))
    parts.append(f"[本次新增 {len(added)} 条，已存在 {len(dup)} 条]")
    return "\n".join(parts)


def find_official_careers(company):
    """针对指定公司搜官方招聘页，结果同样去重入库。"""
    try:
        response = _search_web(f"{company} 招聘 官网 careers jobs")
    except SearchUnavailable as e:
        return str(e)
    except RuntimeError as e:
        return f"错误：{e}"
    except Exception as e:
        logger.exception("搜索失败")
        return f"搜索失败（{type(e).__name__}）：无法获取最新招聘信息。"

    official, _ = _filter_official(response.get("results", []) or [])
    if not official:
        return f"未找到 {company} 的官方招聘页。"

    q = f"{company} 官方招聘"
    added, dup = _save_jobs(_extract_fields(official) or _parse_jobs(official, q),
                            query=q)
    lines = [f"- {r.get('title')}\n  {r.get('url')}" for r in official]
    return (f"{company} 官方招聘页：\n" + "\n".join(lines)
            + f"\n[本次新增 {len(added)} 条，已存在 {len(dup)} 条]")


# ---------- 工具定义 ----------
def _tool(name, desc, props=None, required=()):
    params = {"type": "object", "properties": props or {}}
    if required:
        params["required"] = list(required)
    return {"type": "function",
            "function": {"name": name, "description": desc, "parameters": params}}


def _p(type_, desc, **extra):
    return {"type": type_, "description": desc, **extra}


tools = [
    _tool("find_job",
          "联网搜索岗位，自动过滤招聘聚合站、自动去重并写入记忆。"
          "只返回本次新发现的岗位；已记录过的会单独列出。",
          {"query": _p("string", "搜索关键词，例如『武汉 agent 开发 招聘』")},
          ["query"]),
    _tool("find_official_careers",
          "针对指定公司，搜索其企业官方招聘页（过滤聚合站，结果写入记忆）。",
          {"company": _p("string", "公司名称")}, ["company"]),
    _tool("get_profile", "读取用户画像（城市、年龄、技能等），不包含岗位记录。"),
    _tool("save_memory", "保存用户基本信息，如年龄、城市、技能、喜好等。",
          {"key": _p("string", "信息名称"),
           "value": _p("string", "信息内容。append 模式下多个值用「、」分隔"),
           "mode": _p("string", "标量用 overwrite，列表用 append",
                      enum=["overwrite", "append"])},
          ["key", "value"]),
    _tool("save_job", "把一条具体岗位存入记忆（自动去重）。",
          {"title": _p("string", "岗位名称"),
           "company": _p("string", "公司，未知留空"),
           "location": _p("string", "地点"),
           "source": _p("string", "来源链接"),
           "query": _p("string", "相关搜索词")}, ["title"]),
    _tool("list_jobs", "列出记忆里已保存的岗位。",
          {"limit": _p("integer", "最多返回条数，默认 20")}),
]

TOOL_FUNCS = {"find_job": find_job,
              "find_official_careers": find_official_careers,
              "get_profile": get_profile,
              "save_memory": save_memory,
              "save_job": save_job,
              "list_jobs": list_jobs}


# ---------- Agent 主循环 ----------
AGENT_SYSTEM_PROMPT = """
你是一个带记忆的求职助手。

【记忆规则】
- 用户提问中包含年龄、城市/地点、喜好、职业、技能、学历、薪资期望等信息时，
  必须先调用 save_memory 保存，再执行其他任务。
- 标量信息（城市、年龄）用 mode="overwrite"；列表信息（技能、喜好）用
  mode="append"，多个值用「、」分隔。
- 用户询问自己的信息时，调用 get_profile 读取后再回答。
- 不要编造没保存过的信息。

【岗位记忆规则】
- 调用 find_job / find_official_careers 后，搜索结果会自动过滤、去重并存入岗位记忆。
- 只返回"本次新发现"的岗位；已记录过的会单独列出，不会重复展示。
- 用户明确提到某个具体岗位/公司时，调用 save_job 记录。
- 用户询问"我之前找过哪些工作"时，调用 list_jobs 读取。
- 记忆中的岗位若公司/地点为空，不得凭空补全。

【求职规则】
- 优先返回企业官方招聘页（careers./jobs. 子域，或路径含 /careers、/jobs、/join）。
- 不返回招聘聚合站（BOSS直聘、智联、51job、猎聘、拉勾、脉脉等）。
- 若没有官网结果，如实说明"未找到官方招聘页"，可建议用户提供公司名，
  改用 find_official_careers 再搜。
- 用户指定公司名时，优先调用 find_official_careers 而非 find_job。
- 无法获取最新信息时，明确说明来源限制，不得编造岗位、公司或招聘状态。

【输出格式】
- 只能通过调用工具完成任务，不要输出 Thought/Action 文本。
- 信息足够时直接给出最终答案（不再调用工具）。
"""


def _profile_text():
    profile = load_memory().get("profile", {})
    return ("当前用户画像：" + json.dumps(profile, ensure_ascii=False)
            if profile else "当前用户画像：暂无。")


def _remember_turn(history, user_input, answer):
    """写入一轮对话，并裁剪到最近 MAX_HISTORY_TURNS 轮。

    旧版 history 无上限，聊得越久每轮请求就越贵，最后直接超出上下文窗口。
    """
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": answer})
    overflow = len(history) - MAX_HISTORY_TURNS * 2
    if overflow > 0:
        del history[:overflow]


def agent(user_input, history=None, max_steps=MAX_STEPS):
    """跑一轮对话。

    history：由调用方持有的多轮对话列表，形如
             [{"role":"user",...},{"role":"assistant",...}, ...]
             传同一个列表即可获得跨轮上下文。
    """
    history = history if history is not None else []
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "system", "content": _profile_text()}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    for step in range(max_steps):
        try:
            response = get_client().chat.completions.create(
                model=MODEL, messages=messages, tools=tools,
                tool_choice="auto", temperature=0.2, timeout=60)
        except Exception as e:
            logger.exception("调用模型失败")
            return f"调用模型失败：{type(e).__name__}: {e}"

        msg = response.choices[0].message
        logger.info("[step %d] tool_calls=%s content=%s", step,
                    [tc.function.name for tc in (msg.tool_calls or [])],
                    (msg.content or "")[:100])

        if not msg.tool_calls:
            answer = msg.content or "（模型没有返回内容）"
            _remember_turn(history, user_input, answer)
            return answer

        messages.append(msg.model_dump(exclude_none=True))

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            func = TOOL_FUNCS.get(name)
            if not callable(func):
                result = f"未知工具: {name}"
            else:
                try:
                    result = func(**args)
                except TypeError as e:
                    result = f"参数错误: {e}"          # 参数纠错回灌模型
                except Exception as e:
                    logger.exception("工具 %s 执行异常", name)
                    result = f"执行异常: {type(e).__name__}: {e}"

            content = str(result)
            if len(content) > TOOL_RESULT_LIMIT:
                content = content[:TOOL_RESULT_LIMIT] + "\n...(内容过长已截断)"
            logger.info("[step %d] %s(%s) -> %s", step, name, args, content[:120])

            messages.append({"role": "tool",
                             "tool_call_id": tool_call.id, "content": content})

    mem = load_memory()
    return (f"抱歉，我在 {max_steps} 步内未能完成该请求。\n"
            f"当前用户画像：{json.dumps(mem.get('profile', {}), ensure_ascii=False)}\n"
            f"已记录岗位数：{len(mem.get('jobs', []))}")


# ---------- 数据清洗 ----------
def clean_memory(dry_run=False):
    """移除聚合站来源与重复岗位。

    dry_run=True 时只报告、不写盘；备份带时间戳写入 backups/，可反复安全执行。
    """
    jobs = load_memory().get("jobs") or []
    if not jobs:
        print("记忆里没有岗位，无需清洗。")
        return

    kept, dropped, seen = [], [], set()
    for j in jobs:
        if _is_aggregator(j.get("source", "")) or _has_brand_noise(j.get("title")):
            dropped.append(("聚合站", j.get("title", "")))
            continue
        k = _job_key(j)
        if k and k in seen:
            dropped.append(("重复", j.get("title", "")))
            continue
        if k:
            seen.add(k)
        kept.append(j)

    print(f"岗位记录：{len(jobs)} 条 → 保留 {len(kept)} 条，清理 {len(dropped)} 条")
    for why, title in dropped:
        print(f"  [{why}] {(title or '')[:50]}")
    if dry_run:
        print("（--dry-run：未写入任何文件）")
        return

    backup = _backup_mem_file(tag="clean")
    mem = load_memory()
    mem["jobs"] = kept
    mem["schema_version"] = SCHEMA_VERSION
    _atomic_write(mem, backup=False)
    print(f"已写回 {os.path.basename(MEM_FILE)}")
    if backup:
        print(f"清洗前备份：{os.path.relpath(backup, BASE_DIR)}")


# ---------- 离线自测 ----------
def selftest():
    """纯离线自测：不联网、不需要密钥、不碰真实数据文件。

    会把 MEM_FILE / BACKUP_DIR 临时指向沙箱目录，
    跑完恢复原值并删除沙箱，真实数据一个字节都不会被改动。
    """
    passed, failed = [], []

    def check(name, cond, detail=""):
        (passed if cond else failed).append((name, detail) if not cond else name)
        mark = "✅" if cond else "❌"
        print(f"  {mark} {name}" + ("" if cond or not detail else f"   → {detail}"))

    def raises(fn):
        """断言这个调用「就应该报错」。"""
        try:
            fn()
        except Exception:
            return True
        return False

    print("=" * 62)
    print("离线自测（不联网 / 不需要密钥 / 不碰真实数据）")
    print("=" * 62)

    global MEM_FILE, BACKUP_DIR
    saved = (MEM_FILE, BACKUP_DIR)
    sandbox = tempfile.mkdtemp(prefix="mem-selftest-")
    MEM_FILE = os.path.join(sandbox, "job_prefs.json")
    BACKUP_DIR = os.path.join(sandbox, BACKUP_DIRNAME)

    root_logger = logging.getLogger()
    old_level = root_logger.level
    root_logger.setLevel(logging.CRITICAL)      # 沙箱里会故意造损坏文件，别刷警告
    try:
        print("\n[1] 文本归一化")
        got = _normalize("[职位信息]AI Agent工程师- 应届生求职网")
        check("删掉噪声括号 [职位信息] 与尾部「- 应届生求职网」",
              got == "aiagent工程师", repr(got))
        got = _normalize("Agent开发工程师-BOSS直聘")
        check("删掉尾部「-BOSS直聘」", got == "agent开发工程师", repr(got))
        got = _normalize("【Agent开发工程师】")
        check("括号里是岗位名时不会被连内容一起删", got == "agent开发工程师", repr(got))

        print("\n[2] 去重键")
        check("同一 URL 的不同写法撞键",
              _job_key({"source": "https://A.com/job/1", "title": "甲"})
              == _job_key({"source": "http://a.com/job/1/", "title": "乙"}))
        check("无 URL 退回「公司+岗位+地点」",
              (_job_key({"company": "某某科技", "title": "Agent",
                         "location": "武汉"}) or ("",))[0] == "C")
        check("只有标题时用 T 键", (_job_key({"title": "Agent"}) or ("",))[0] == "T")
        check("空记录返回 None", _job_key({}) is None)

        print("\n[3] 聚合站与官网判定")
        check("58.com 命中聚合站", _is_aggregator("https://www.58.com/x"))
        check("158.com 不被误杀", not _is_aggregator("https://158.com/x"))
        check("x51job.com 不被误杀", not _is_aggregator("https://x51job.com/x"))
        check("zhipin.com 子域命中", _is_aggregator("https://www.zhipin.com/job/1"))
        check("careers 子域识别为官方招聘页",
              _looks_like_career_page("https://careers.tencent.com/x"))
        check("普通介绍页不被误判为招聘页",
              not _looks_like_career_page("https://www.tencent.com/about"))

        print("\n[4] 画像记忆")
        save_memory("技能", "Python、Java，SQL;CET-4", mode="append")
        save_memory("技能", "Python", mode="append")
        skills = load_memory()["profile"].get("技能")
        check("多值拆分 + 去重", skills == ["Python", "Java", "SQL", "CET-4"], skills)

        print("\n[5] 岗位写入与去重")
        j1 = {"title": "Agent工程师", "company": "某某科技", "location": "武汉",
              "source": "https://careers.x.com/jobs/1"}
        added, dup = _save_jobs([dict(j1)], query="武汉 agent 招聘")
        check("首次写入 1 条", len(added) == 1 and not dup)
        added, dup = _save_jobs([dict(j1)], query="武汉 智能体 招聘")
        check("同一岗位不再重复写入", not added and len(dup) == 1)
        check("无新岗位时搜索词照样落盘",
              "武汉 智能体 招聘" in load_memory()["queries"])

        print("\n[6] 重复岗位字段补全")
        _save_jobs([{"title": "数据分析师", "location": "",
                     "source": "https://careers.y.com/jobs/2"}])
        _save_jobs([{"title": "数据分析师", "company": "某某数据", "location": "杭州",
                     "source": "https://careers.y.com/jobs/2"}])
        rows = [j for j in load_memory()["jobs"] if "y.com" in (j.get("source") or "")]
        check("旧记录缺失的公司/地点被补上",
              len(rows) == 1 and rows[0].get("company") == "某某数据"
              and rows[0].get("location") == "杭州", rows)

        print("\n[7] 数据安全")
        check("拒绝写入空内容", raises(lambda: _atomic_write({})))

        shutil.copy(MEM_FILE, MEM_FILE + ".bak")
        open(MEM_FILE, "w", encoding="utf-8").close()          # 清成 0 字节
        check("主文件被清空后能沿回退链读回数据",
              len(load_memory()["jobs"]) > 0)

        with open(MEM_FILE, "w", encoding="utf-8") as f:
            f.write("{这不是合法 json")
        check("损坏的主文件不会静默返回空记忆", len(load_memory()["jobs"]) > 0)

        os.remove(MEM_FILE)
        check("主文件被删除时同样能读回", len(load_memory()["jobs"]) > 0)

        _atomic_write(load_memory())
        check("写入后主文件被自动修复",
              os.path.exists(MEM_FILE) and os.path.getsize(MEM_FILE) > 0)

        print("\n[8] 备份与恢复")
        _backup_mem_file(tag="selftest")
        check("备份已写入 backups/", len(list_backups()) >= 1, list_backups())

        print("\n[9] 多轮上下文裁剪")
        h = []
        for i in range(MAX_HISTORY_TURNS + 5):
            _remember_turn(h, f"u{i}", f"a{i}")
        check(f"history 裁剪到 {MAX_HISTORY_TURNS} 轮",
              len(h) == MAX_HISTORY_TURNS * 2, len(h))
        check("保留的是最近的对话", h[-1]["content"] == f"a{MAX_HISTORY_TURNS + 4}")

        print("\n[10] 岗位条数上限")
        many = [{"title": f"岗位{i}", "location": "", "company": "",
                 "source": f"https://z.com/jobs/{i}",
                 "found_at": f"2026-01-01T00:00:{i % 60:02d}"}
                for i in range(MAX_JOBS + 10)]
        check(f"超出 {MAX_JOBS} 条时淘汰最旧",
              len(_trim_jobs(many)) == MAX_JOBS, len(_trim_jobs(many)))
    finally:
        root_logger.setLevel(old_level)
        MEM_FILE, BACKUP_DIR = saved
        shutil.rmtree(sandbox, ignore_errors=True)

    print("\n" + "-" * 62)
    if failed:
        print(f"通过 {len(passed)} 项，失败 {len(failed)} 项：")
        for name, detail in failed:
            print(f"  ❌ {name}" + (f"   → {detail}" if detail else ""))
    else:
        print(f"全部通过 ✅（共 {len(passed)} 项）")
    print("-" * 62)
    return not failed


# ---------- 入口 ----------
def demo():
    history = []
    print("=== 第一次搜索 ===")
    print(agent("我在武汉找三个跟agent开发有关的工作", history=history))
    print("\n=== 第二次搜索（同样的词，验证不重复）===")
    print(agent("再帮我找找武汉 agent 开发的工作", history=history))
    print("\n=== 查看已记录岗位 ===")
    print(agent("我之前找过哪些工作？", history=history))


def interactive():
    history = []
    print("求职助手已启动。输入 exit / q 退出，输入 /reset 清空对话上下文。")
    print("跑不动时先执行：python agent.py diag（一键区分代码问题 / 网络问题）")
    while True:
        try:
            q = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            break
        if q == "/reset":
            history.clear()
            print("对话上下文已清空（长期记忆保留）。")
            continue
        try:
            print("\n助手：", agent(q, history=history))
        except Exception as e:
            logger.exception("对话失败")
            print(f"\n助手：出错了（{type(e).__name__}: {e}）")


def _cmd_backups():
    rows = list_backups()
    if not rows:
        print(f"（{BACKUP_DIRNAME}/ 里没有备份）")
        return
    print(f"共 {len(rows)} 份备份（新 → 旧）：")
    for p in rows:
        print(f"  {os.path.relpath(p, BASE_DIR)}   {os.path.getsize(p)} 字节")


COMMANDS = {
    "selftest": lambda: sys.exit(0 if selftest() else 1),
    "diag": diag,
    "demo": demo,
    "backups": _cmd_backups,
    "clean": lambda: clean_memory(dry_run="--dry-run" in sys.argv[2:]),
    "restore": lambda: sys.exit(0 if restore(
        sys.argv[2] if len(sys.argv) > 2 else None) else 1),
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in ("-h", "--help", "help"):
        print(__doc__)
    elif cmd in COMMANDS:
        COMMANDS[cmd]()
    elif cmd:
        print(f"未知命令：{cmd}\n可用命令：{' / '.join(COMMANDS)}（无参数 = 交互式对话）")
    else:
        interactive()
