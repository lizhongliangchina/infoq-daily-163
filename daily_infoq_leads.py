
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, datetime as dt, zoneinfo, logging
import re, math, yaml, json, smtplib, hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, urljoin
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests
from bs4 import BeautifulSoup

def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def canonicalize_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()
        path = parsed.path or "/"
        query_pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                       if not k.lower().startswith(("utm_", "fbclid", "gclid", "igshid", "mc_cid", "mc_eid"))]
        query = urlencode(query_pairs, doseq=True)
        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return url

def text_clean(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '').strip())

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def time_decay(pub_time: Optional[dt.datetime], hours: float) -> float:
    if not pub_time:
        return 1.0
    delta = now_utc() - pub_time
    tau = dt.timedelta(hours=hours)
    x = max(delta.total_seconds(), 0.0) / max(tau.total_seconds(), 1.0)
    return math.exp(-x)

def pos_score(idx: int) -> float:
    return 1.0 / math.log2(3 + idx)

def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

@dataclass
class Item:
    source: str
    title: str
    url: str
    pub_time: Optional[dt.datetime] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    heat: float = 0.0
    relevance: float = 0.0
    velocity: float = 0.0
    corroboration: float = 0.0
    originality: float = 0.0
    practicality: float = 0.0
    impact: float = 0.0
    depth: float = 0.0
    novelty: float = 0.0
    authority: float = 0.0
    diversity_bonus: float = 0.0
    fatigue: float = 0.0
    spam_risk: float = 0.0
    categories: List[str] = field(default_factory=list)
    angles: List[str] = field(default_factory=list)
    score: float = 0.0

class RelevanceScorer:
    def __init__(self, keywords_cfg: Dict[str, Any]):
        self.kw_map = {cat: [kw.lower() for kw in kws] for cat, kws in keywords_cfg.get("categories", {}).items()}
        self.cat_weights = keywords_cfg.get("category_weights", {})
        self.angle_bank = keywords_cfg.get("angle_bank", {})

    def score_and_tag(self, title: str, text: str = "") -> Tuple[float, List[str]]:
        s = f"{title} {text}".lower()
        score = 0.0
        tags = []
        for cat, kws in self.kw_map.items():
            hits = sum(1 for kw in kws if kw in s)
            if hits:
                w = float(self.cat_weights.get(cat, 1.0))
                score += w * min(hits, 3)
                tags.append(cat)
        return score, tags

    def suggest_angles(self, tags: List[str]) -> List[str]:
        angles = []
        for t in tags[:2]:
            angles.extend(self.angle_bank.get(t, [])[:2])
        angles.extend(self.angle_bank.get("_general_", [])[:2])
        seen, out = set(), []
        for a in angles:
            if a not in seen:
                seen.add(a); out.append(a)
        return out[:4]

def safe_get(url: str, timeout: int = 15):
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        return r if r.status_code == 200 else None
    except Exception:
        return None

def fetch_generic(url: str, source: str, selector: str, max_items: int=40):
    resp = safe_get(url)
    if not resp: return []
    soup = BeautifulSoup(resp.text, "html.parser")
    nodes = soup.select(selector)
    items = []
    for i, a in enumerate(nodes[:max_items]):
        href = a.get("href") or ""
        txt = text_clean(a.get_text())
        if not href or not txt: continue
        href = canonicalize_url(urljoin(url, href))
        items.append(Item(source=source, title=txt, url=href, heat=1.0/(1+0.3*i), meta={"rank": i+1}))
    return items

def fetch_hn(url: str, max_items: int = 30) -> List[Item]:
    resp = safe_get(url)
    if not resp: return []
    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("tr.athing")
    items = []
    for i, r in enumerate(rows[:max_items]):
        a = r.select_one("span.titleline a")
        if not a: continue
        href = canonicalize_url(a["href"])
        title = text_clean(a.get_text())
        meta_row = r.find_next_sibling("tr")
        score_el = meta_row.select_one("span.score") if meta_row else None
        comments_el = meta_row.find("a", string=re.compile(r"\bcomments?\b|\bcomment\b")) if meta_row else None
        points = int(re.search(r"(\d+)", score_el.get_text()).group(1)) if score_el else 0
        comments = int(re.search(r"(\d+)", comments_el.get_text()).group(1)) if comments_el else 0
        items.append(Item(source="HackerNews", title=title, url=href, meta={"points": points, "comments": comments, "rank": i+1}, heat=points + 0.5*comments))
    return items

def fetch_techcrunch(u):  return fetch_generic(u, "TechCrunch", "a.post-block__title__link", 40)
def fetch_theverge(u):    return fetch_generic(u, "TheVerge", "h2 a, h3 a", 40)
def fetch_slashdot(u):    return fetch_generic(u, "Slashdot", "article .story-title a", 40)
def fetch_engadget(u):    return fetch_generic(u, "Engadget", "h2 a, h3 a", 40)
def fetch_arstechnica(u): return fetch_generic(u, "ArsTechnica", "h2 a, h3 a", 40)

def fetch_techmeme(url: str) -> List[Item]:
    resp = safe_get(url)
    if not resp: return []
    soup = BeautifulSoup(resp.text, "html.parser")
    anchors = soup.select("div#river a") or soup.select("a[href]")
    seen = set(); out = []
    for i, a in enumerate(anchors):
        href = a.get("href") or ""
        txt = text_clean(a.get_text())
        if not href or not txt: continue
        href = canonicalize_url(urljoin(url, href))
        key = (txt, href)
        if key in seen: continue
        seen.add(key)
        if "techmeme.com" in extract_domain(href): continue
        out.append(Item(source="Techmeme", title=txt, url=href, heat=1.0/(1+0.3*i), meta={"rank": i+1}))
        if len(out)>=40: break
    return out

OFFICIAL_DOMAINS_HINTS = ["github.com","arxiv.org","research.","docs.","developers.","cloud.google.com","openai.com","anthropic.com","meta.com","ai.google","deepmind.com","huggingface.co","apache.org","k8s.io","pytorch.org","tensorflow.org","nvidia.com","intel.com","amd.com","rust-lang.org","go.dev","python.org","nodejs.org","mozilla.org"]
AUTHORITATIVE_SITES = ["techmeme.com","arstechnica.com","techcrunch.com","theverge.com","engadget.com","slashdot.org","news.ycombinator.com"]

def authority_score(url: str, whitelist: list, blacklist: list) -> float:
    d = extract_domain(url)
    if any(b in d for b in blacklist): return 0.2
    if d in whitelist: return 1.0
    if any(h in url for h in OFFICIAL_DOMAINS_HINTS): return 1.0
    if d in AUTHORITATIVE_SITES: return 0.8
    return 0.6

def originality_score(url: str, whitelist: list) -> float:
    d = extract_domain(url)
    if d in whitelist or any(h in url for h in OFFICIAL_DOMAINS_HINTS): return 1.0
    return 0.6

def cluster_items(items: List[Item]) -> List[List[Item]]:
    buckets = {}
    for it in items:
        key = canonicalize_url(it.url)
        key2 = re.sub(r'[^a-z0-9]+','', it.title.lower())
        cluster_key = sha1(extract_domain(key) + "|" + key2[:60])
        buckets.setdefault(cluster_key, []).append(it)
    return list(buckets.values())

def choose_representative(cluster: List[Item], whitelist: list) -> Item:
    def prio(it: Item):
        d = extract_domain(it.url)
        w = 0 if (d in whitelist or any(h in it.url for h in OFFICIAL_DOMAINS_HINTS)) else (1 if "techmeme.com" in d else 2)
        return (w, -(it.heat + it.authority))
    return sorted(cluster, key=prio)[0]

def compute_score(it: Item, w: Dict[str, float], decay_hours: float) -> float:
    decay = time_decay(it.pub_time, decay_hours)
    s = (w["heat"]*it.heat + w["relevance"]*it.relevance + w["velocity"]*it.velocity + w["corroboration"]*it.corroboration +
         w["originality"]*it.originality + w["practicality"]*it.practicality + w["impact"]*it.impact + w["depth"]*it.depth +
         w["novelty"]*it.novelty + w["authority"]*it.authority + w["diversity_bonus"]*it.diversity_bonus)
    s = s * decay - w["fatigue"]*it.fatigue - w["spam_risk"]*it.spam_risk
    return s

def apply_bucket_constraints(items: List[Item], buckets_cfg: Dict[str, Dict[str, int]]) -> List[Item]:
    counts = {k:0 for k in buckets_cfg}
    out = []
    for it in items:
        tgt = None
        for b in buckets_cfg.keys():
            if b in it.categories:
                tgt = b; break
        if not tgt:
            out.append(it); continue
        cap = buckets_cfg[b].get("max", 999)
        if counts[b] < cap:
            out.append(it); counts[b]+=1
    return out

def mmr_select(items: List[Item], k: int=10, lamb: float=0.7) -> List[Item]:
    def sim(a: str, b: str) -> float:
        ta = set(re.findall(r"[a-z0-9]+", a.lower())); tb = set(re.findall(r"[a-z0-9]+", b.lower()))
        return len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    chosen, cand = [], items[:]
    while cand and len(chosen) < k:
        best, best_score = None, -1e9
        for it in cand:
            rel = it.score
            div = 0.0 if not chosen else max(sim(it.title, c.title) for c in chosen)
            mmr = lamb * rel - (1 - lamb) * div
            if mmr > best_score: best, best_score = it, mmr
        chosen.append(best); cand.remove(best)
    return chosen

def build_email_html(subject: str, items: List[Item]) -> str:
    rows = []
    for i, it in enumerate(items, 1):
        tags = " / ".join(it.categories) if it.categories else "General"
        angles = "；".join(it.angles)
        rows.append(f"""
        <tr>
          <td style="padding:8px 6px; vertical-align:top;">{i}</td>
          <td style="padding:8px 6px;">
            <div style="font-weight:600; font-size:15px;">
              <a href="{it.url}" target="_blank" rel="noopener">{it.title}</a>
            </div>
            <div style="font-size:12px; color:#444; margin-top:4px;">来源：{it.source} ｜ 标签：{tags}</div>
            <div style="font-size:12px; color:#111; margin-top:6px;">建议选题方向：{angles}</div>
          </td>
        </tr>
        """)
    table = "\n".join(rows)
    html = f"""
    <html><body style="font-family:Arial,Helvetica,sans-serif;">
      <h3 style="margin:0 0 10px 0;">InfoQ 日报线索 · {dt.datetime.now().strftime('%Y-%m-%d')}</h3>
      <table border="0" cellspacing="0" cellpadding="0" width="100%">{table}</table>
      <div style="margin-top:16px; font-size:12px; color:#666;">说明：仅提供“线索链接+主题标签+建议角度”。</div>
    </body></html>"""
    return html

def send_via_smtp(cfg: Dict[str, Any], subject: str, html: str) -> None:
    smtp_cfg = cfg["email"]["smtp"]
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = cfg["email"]["from_email"]
    msg["To"] = ", ".join(cfg["email"]["to_emails"])
    msg.attach(MIMEText(html, "html", "utf-8"))

    host = smtp_cfg["host"]
    port = smtp_cfg.get("port", 587)
    user = smtp_cfg["user"]
    # 优先用环境变量，避免把授权码写进仓库
    pwd  = os.environ.get("SMTP_PASSWORD") or smtp_cfg.get("password")
    if not pwd:
        raise RuntimeError("SMTP password not provided; set env SMTP_PASSWORD or config email.smtp.password")

    use_tls = smtp_cfg.get("use_tls", True)
    use_ssl = smtp_cfg.get("use_ssl", False)

    if use_ssl or port == 465:
        server = smtplib.SMTP_SSL(host, port)
        server.login(user, pwd)
    else:
        server = smtplib.SMTP(host, port)
        if use_tls:
            server.starttls()
        server.login(user, pwd)

    server.sendmail(cfg["email"]["from_email"], cfg["email"]["to_emails"], msg.as_string())
    server.quit()

def harvest(cfg: Dict[str, Any], rel) -> List[Item]:
    items: List[Item] = []
    sources = cfg["sources"]
    for u in sources:
        if "news.ycombinator.com" in u: items.extend(fetch_hn(u))
        elif "techcrunch.com" in u: items.extend(fetch_techcrunch(u))
        elif "theverge.com" in u: items.extend(fetch_theverge(u))
        elif "slashdot.org" in u: items.extend(fetch_slashdot(u))
        elif "engadget.com" in u: items.extend(fetch_engadget(u))
        elif "arstechnica.com" in u: items.extend(fetch_arstechnica(u))
        elif "techmeme.com" in u: items.extend(fetch_techmeme(u))
    wl = set(cfg.get("whitelist_domains", []))
    bl = set(cfg.get("blacklist_domains", []))
    for it in items:
        rel_score, tags = rel.score_and_tag(it.title)
        it.relevance = rel_score; it.categories = tags
        it.authority = authority_score(it.url, wl, bl)
        it.originality = originality_score(it.url, wl)
        if re.search(r"\b(how|guide|benchmark|case|postmortem|lessons|architecture)\b", it.title.lower()):
            it.practicality = 1.0; it.depth = 0.6
        it.impact = 1.0 if any(k in it.title.lower() for k in ("kubernetes","openai","anthropic","nvidia","chrome","pytorch","tensorflow","security")) else 0.5
        it.novelty = 0.6; it.spam_risk = 0.0
    return items

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 8:00 America/Los_Angeles gate (用于 GitHub Actions / 解决夏令时问题)
    if os.environ.get("GATE_PACIFIC_8AM", "false").lower() in ("1","true","yes"):
        la = dt.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
        if la.hour != 8:
            print(f"[gate] Now in LA is {la.isoformat()}, not 08:00 hour, exit without sending.")
            sys.exit(0)

    cfg_path = os.environ.get("INFOQ_CFG", "config.yaml")
    cfg = load_yaml(cfg_path)
    kw = load_yaml(cfg.get("keywords_path", "keywords.yaml"))
    rel = RelevanceScorer(kw)

    items = harvest(cfg, rel)
    clusters = cluster_items(items)
    reps = []
    for cl in clusters:
        for it in cl:
            it.corroboration = min(1.0, (len(cl) - 1) / 3.0)
        reps.append(choose_representative(cl, cfg.get("whitelist_domains", [])))

    w = cfg["weights"]
    for it in reps:
        it.angles = rel.suggest_angles(it.categories)
        it.score = compute_score(it, w, cfg.get("time_decay_hours", 18))

    reps.sort(key=lambda x: x.score, reverse=True)
    constrained = apply_bucket_constraints(reps, cfg.get("buckets", {}))
    topk = mmr_select(constrained, k=int(cfg.get("topk", 10)), lamb=cfg.get("mmr_lambda", 0.7))

    subject = cfg.get("email", {}).get("subject_prefix", "InfoQ 日报线索") + " - " + dt.datetime.now().strftime("%Y-%m-%d")
    html = build_email_html(subject, topk)

    method = cfg.get("email", {}).get("method", "smtp")
    if method == "smtp":
        send_via_smtp(cfg, subject, html)
    else:
        raise RuntimeError("For 163 email, set email.method=smtp in config.yaml")
    logging.info("Sent daily leads: %d items", len(topk))

if __name__ == "__main__":
    main()
