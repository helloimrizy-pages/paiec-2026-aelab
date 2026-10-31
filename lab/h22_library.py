"""H22 public-results library: text-matchable index of per-item, per-model outcomes from inventory benchmarks.

Input: data/public_results/<slug>/{items,results}.parquet, one folder per benchmark (built by tools/build_library.py),
plus READY.txt listing the slugs to include. items: item_key, text, extra (JSON string).
results: item_key, model (name as released), config (harness/setting or ""), trial (1-based), outcome (1.0 correct /
0.0 incorrect, only grades explicit in the source).
Model names are mapped to measurement-db canonical names via measurement-db-code's map_model_registry.json,
which is how the hidden `normalized_name` field is produced.

Matching: every text value of an item is tokenized ([a-z0-9]+); the library stores hashes of up to K evenly spaced
8-token shingles per item; a query hashes ALL 8-token shingles of every text value of the hidden item_content
(a JSON wrapper is parsed) and votes. Robust to added prefixes/suffixes, templates and JSON wrapping.

  python h22_library.py build      -> features/h22_lib/{shingles.npy,items.parquet,outcomes.parquet}
  python h22_library.py selftest   -> wrapped-text recall and false-positive rate on measurement-db items
"""
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "public_results"
LIB = Path(__file__).resolve().parent / "features" / "h22_lib"
REG = ROOT / "measurement-db-code" / "scripts" / "build_measurement_tables" / "map_model_registry.json"
EXCLUDE_SUBMISSIONS = {"swe_bench_live": ["slai"]}
# Benchmarks kept out of the runtime library: item identity is not in the text, or redistribution is not allowed.
EXCLUDE_SLUGS = {"capability": "11,994 items share 2 prompt texts (identity is the image/video); GitHub license forbids redistribution",
                 "medcalc_bench": "only v1.0 outputs (gpt-4o-mini, partial gpt-4/3.5); v1.2/Verified changed most notes, questions and answers, so a shared note could inherit a result graded against a different answer; no 2025-26 model results",
                 "kor_bench": "no 2025-26 per-item results known anywhere (62 models, 2023-24); 45 operation items have a released grading bug (LaTeX spacing)",
                 "erbench": "no 2025-26 model results (unlikely hidden); 204k templated items with many duplicate texts would grow the library ~10x",
                 "os_harm": "outcome direction ambiguous: ingested 1 = safe, but measurement-db codes safety benchmarks by their native metric (OS-Harm reports unsafe rate); an inverted lookup cannot be corrected by labels"}   # success-only package (all rows 1.0), see its README
# Benchmarks where identical item text means the same task (e.g. one task across benchmark versions, or one problem
# cross-listed in two competitions): such items are merged into one library item with pooled outcomes. Elsewhere
# (e.g. MathVista, where the image differs) identical texts stay separate and the tie guard declines them.
MERGE_DUPLICATE_TEXT = {"terminal_bench", "open_proof_corpus", "osworld"}
# Same task under a re-worded prompt (item_key suffix): merged into the base item with pooled outcomes.
MERGE_KEY_SUFFIX = {"cybench": r"#pv2$"}
# Row flags (results.parquet `flags`) that exclude a row: a disputed model label would corrupt exact subject lookups.
EXCLUDE_FLAGS = ["label_disputed"]
# Model strings that are not single runs of a subject (e.g. best-of-8 selections of another model's outputs).
EXCLUDE_MODELS = {"open_proof_corpus": r"Judge\)"}
# Items whose released pass rule is ambiguous (two opposite rules in the release): excluded so a lookup cannot invert them.
# The whole ICF-Bench dynamic_preference scenario is excluded: its noforget text is contained in the excluded forget text,
# so a hidden forget item would otherwise match the noforget library item and inherit the wrong setting's outcomes.
EXCLUDE_ITEMS = {"icf_bench": r"^(dynamic_preference|subtask_revision)/"}   # subtask_revision forget text holds a per-model placeholder
# Manual canonical names for release-specific model strings the registry cannot map (checked against registry values).
MANUAL_ALIASES = {"o3Mini-Eval": "OpenAI o3-mini", "gpt4o1120-Eval": "OpenAI GPT-4o",
                  "gpt-5-chat-shortco-2025-08-07-Eval": "OpenAI GPT-5 Chat", "gpt-5-chat-shortco-2025-08-07-Bing": "OpenAI GPT-5 Chat",
                  "deepseek/r1_0528": "DeepSeek R1"}
W, K = 8, 24                      # shingle width (tokens), shingles stored per library item
TOK = re.compile(r"[a-z0-9]+(?:\+\+|#)?")   # keeps c++ / c# / f# distinct


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)


def texts(content) -> list[str]:
    """All string leaves of a JSON item_content (or the raw string)."""
    s = "" if content is None else str(content)
    out = [s]
    try:
        stack = [json.loads(s)]
        while stack:
            o = stack.pop()
            if isinstance(o, str): out.append(o)
            elif isinstance(o, dict): stack.extend(o.values())
            elif isinstance(o, list): stack.extend(o)
    except Exception:
        pass
    return out


SHORT_MIN = 5                     # short texts (SHORT_MIN..W-1 tokens) are indexed whole


def shingles(tokens: list[str]) -> list[str]:
    if len(tokens) < W:
        return [" ".join(tokens)] if len(tokens) >= SHORT_MIN else []
    return [" ".join(tokens[i:i + W]) for i in range(len(tokens) - W + 1)]


def lib_shingles(text: str, freq=None) -> list[int]:
    """Up to K shingle hashes per item. With freq (shingle hash -> number of items of the same benchmark containing it),
    shingles unique to the item are preferred, so near-duplicate items (e.g. one problem in 12 languages) stay distinguishable."""
    hs = list(dict.fromkeys(h64(x) for x in shingles(TOK.findall(str(text).lower()))))
    if len(hs) <= K:
        return hs
    if freq is not None:
        uniq = [h for h in hs if freq.get(h, 0) == 1]
        if uniq:
            pick = [uniq[int(i)] for i in np.linspace(0, len(uniq) - 1, min(K, len(uniq))).round()]
            pick = list(dict.fromkeys(pick))
            rest = [h for h in hs if h not in set(pick)]
            if len(pick) < K and rest:
                pick += [rest[int(i)] for i in np.linspace(0, len(rest) - 1, min(K - len(pick), len(rest))).round()]
            return list(dict.fromkeys(pick))
    return list(dict.fromkeys(hs[int(i)] for i in np.linspace(0, len(hs) - 1, K).round()))


def query_shingles(content) -> np.ndarray:
    """All W-token shingles, plus all n-grams of width SHORT_MIN..W-1 so short library texts can match inside a query."""
    hs = set()
    for t in texts(content):
        tk = TOK.findall(t.lower())
        hs.update(h64(x) for x in shingles(tk))
        for w in range(SHORT_MIN, W):
            hs.update(h64(" ".join(tk[i:i + w])) for i in range(len(tk) - w + 1))
    return np.fromiter(hs, dtype=np.int64)


def canonical_map():
    reg = json.load(open(REG))
    exact = {k: v["model"] for k, v in reg.items() if isinstance(v, dict) and v.get("model")}
    loose = {}
    for k, m in exact.items():
        loose.setdefault(re.sub(r"[^a-z0-9]", "", k.lower()), m)
        loose.setdefault(re.sub(r"[^a-z0-9]", "", m.lower()), m)
    DROP = {"instruct", "chat", "preview", "latest", "hf", "it"}
    def tkey(x):
        return " ".join(sorted(t for t in re.findall(r"[a-z]+|\d+", x.lower()) if t not in DROP))
    bag = {}
    for k, m in exact.items():
        bag.setdefault(tkey(k.split("/")[-1]), m); bag.setdefault(tkey(m), m)
    years = {v["model"]: str(v.get("release_date", ""))[:4] for v in reg.values() if isinstance(v, dict) and v.get("model")}
    exact_last = {}
    for k, m in exact.items():
        exact_last.setdefault(k.split("/")[-1], m)
    ABBR = [(r"(?<![A-Za-z])Instr(?![a-z])", "Instruct"), (r"(?<![A-Za-z])Thk(?![a-z])", "Thinking"),
            (r"(?<![A-Za-z])Rsn(?![a-z])", "Reasoning"), (r"(?<![A-Za-z])Nmt(?![a-z])", "Nemotron")]

    def variants(n: str):
        """Name variants tried in order: as given, abbreviations expanded, effort/serving tags in () or [] removed,
        quantization suffixes removed. The registry merges effort, serving and quantization variants."""
        out = [n]
        e = n
        for pat, rep in ABBR:
            e = re.sub(pat, rep, e)
        out.append(e)
        e2 = re.sub(r"\s*[\(\[][^\)\]]*[\)\]]", "", e).strip()
        out.append(e2)
        out.append(re.sub(r"[-_ ](FP8|FP16|BF16|AWQ|GPTQ|INT4|INT8|GGUF)$", "", e2, flags=re.I))
        return list(dict.fromkeys(v for v in out if v))

    def canon(name: str):
        n = str(name)
        if n in MANUAL_ALIASES: return MANUAL_ALIASES[n]
        for v in variants(n)[1:]:
            if v in exact: return exact[v]
            if v in exact_last: return exact_last[v]
        c = canon_core(n)
        if c: return c
        for inner in re.findall(r"\(([^()]+)\)", n):          # "Claude-3-Opus (claude-3-opus-20240229)" -> try the id,
            inner = inner.strip()                                # exact registry hits only (a loose word-bag match on a
            c = exact.get(inner) or exact_last.get(inner)        # bare date once produced the junk name "my_model/")
            if c: return c
        return None

    def canon_core(n: str):
        for v in variants(n):
            c = canon_one(v)
            if c and not (c.endswith("/") or c.lower().startswith("my_model")):   # reject implausible registry names
                return c
        return None

    def canon_one(n: str):
        if n in exact: return exact[n]
        if n in exact_last: return exact_last[n]
        n2 = n.split("/")[-1]
        if n2 in exact: return exact[n2]
        key = re.sub(r"[^a-z0-9]", "", n2.lower())
        if key in loose: return loose[key]
        key2 = re.sub(r"(20\d{6}|20\d{2}-?\d{2}-?\d{2})$", "", key)   # strip date snapshot suffix
        if key2 in loose: return loose[key2]
        return (bag.get(tkey(n2)) or bag.get(tkey(n)) or bag.get(tkey(re.sub(r"(20\d{6}|20\d{2}-?\d{2}-?\d{2})", "", n2)))
                or bag.get(tkey(re.sub(r"(20\d{6}|20\d{2}-?\d{2}-?\d{2})", "", n))))
    return canon, years


def build():
    canon, years = canonical_map()
    items, outs, sh_h, sh_i = [], [], [], []
    ready = {l.strip() for l in open(SRC / "READY.txt") if l.strip() and not l.startswith("#")}   # reviewed ingestions only
    for d in sorted(p for p in SRC.iterdir() if (p / "items.parquet").exists() and (p / "results.parquet").exists()):
        if d.name not in ready:
            print(f"{d.name}: skipped (not in READY.txt)"); continue
        if d.name in EXCLUDE_SLUGS:
            print(f"{d.name}: excluded ({EXCLUDE_SLUGS[d.name]})"); continue
        it = pd.read_parquet(d / "items.parquet"); rs = pd.read_parquet(d / "results.parquet")
        rs = rs[rs.outcome.notna()]
        if "submission" in rs.columns and d.name in EXCLUDE_SUBMISSIONS:      # biased packages (e.g. success-only)
            rs = rs[~rs.submission.astype(str).str.contains("|".join(EXCLUDE_SUBMISSIONS[d.name]), case=False, regex=True)]
        if d.name in MERGE_KEY_SUFFIX:
            pat = MERGE_KEY_SUFFIX[d.name]
            n0 = rs.item_key.nunique()
            rs = rs.assign(item_key=rs.item_key.astype(str).str.replace(pat, "", regex=True))
            it = it[~it.item_key.astype(str).str.contains(pat, regex=True)]
            print(f"{d.name}: merged re-worded variants, {n0} -> {rs.item_key.nunique()} result items", flush=True)
        outside_texts = []          # items of this benchmark that will NOT be in the library (excluded or without results)
        if d.name in EXCLUDE_ITEMS:
            m = rs.item_key.astype(str).str.contains(EXCLUDE_ITEMS[d.name], regex=True)
            print(f"{d.name}: excluded {int(m.sum())} rows matching {EXCLUDE_ITEMS[d.name]}", flush=True)
            mi = it.item_key.astype(str).str.contains(EXCLUDE_ITEMS[d.name], regex=True)
            outside_texts += it.text[mi].astype(str).tolist()
            rs = rs[~m]; it = it[~mi]
        if d.name in EXCLUDE_MODELS:
            mm = rs.model.astype(str).str.contains(EXCLUDE_MODELS[d.name], regex=True)
            print(f"{d.name}: dropped {int(mm.sum())} rows of selected (not single-run) models", flush=True)
            rs = rs[~mm]
        if "flags" in rs.columns:
            bad = rs["flags"].fillna("").astype(str).str.contains("|".join(EXCLUDE_FLAGS))
            if bad.any(): print(f"{d.name}: dropped {int(bad.sum())} rows flagged {EXCLUDE_FLAGS}", flush=True)
            rs = rs[~bad]
        if d.name in MERGE_DUPLICATE_TEXT:
            canon_key = {}
            first = {}
            for r in it.itertuples():
                t = " ".join(TOK.findall(str(r.text).lower()))
                canon_key[r.item_key] = first.setdefault(t, r.item_key)
            n0 = rs.item_key.nunique()
            rs = rs.assign(item_key=rs.item_key.map(lambda k: canon_key.get(k, k)))
            it = it[it.item_key.isin(set(canon_key.values()))]
            print(f"{d.name}: merged identical texts, {n0} -> {rs.item_key.nunique()} result items", flush=True)
        stats = rs.groupby("item_key").outcome.agg(["mean", "size"])
        base = len(items)
        keyidx = {}
        freq = {}
        for r in it.itertuples():
            if r.item_key in stats.index:
                for h in set(h64(x) for x in shingles(TOK.findall(str(r.text).lower()))):
                    freq[h] = freq.get(h, 0) + 1
        outside_texts += it.text[~it.item_key.isin(stats.index)].astype(str).tolist()
        inv = {}                    # shingle hash -> ids of outside items containing it
        for j, t in enumerate(outside_texts):
            for h in set(h64(x) for x in shingles(TOK.findall(t.lower()))):
                inv.setdefault(h, set()).add(j)
        n_contained = 0
        for r in it.itertuples():
            if r.item_key not in stats.index: continue
            hs = lib_shingles(r.text, freq)
            if not hs: continue
            if inv:
                common = None
                for h in hs:
                    common = inv.get(h, set()) if common is None else common & inv.get(h, set())
                    if not common: break
                if common:          # every stored fragment lies inside an item that is not in the library: a hidden copy
                    n_contained += 1  # of that outside item would match this one, so leave it out
                    continue
            keyidx[r.item_key] = len(items)
            items.append(dict(slug=d.name, item_key=r.item_key, mean=float(stats.loc[r.item_key, "mean"]), n=int(stats.loc[r.item_key, "size"]), n_sh=len(hs)))
            sh_h += hs; sh_i += [keyidx[r.item_key]] * len(hs)
        if n_contained: print(f"{d.name}: left out {n_contained} items contained in non-library items of the same benchmark", flush=True)
        rs = rs[rs.item_key.isin(keyidx)].copy()
        runs = d / "runs.csv"
        if "run_id" in rs.columns and runs.exists():          # prefer the model id actually passed to the agent
            ru = pd.read_csv(runs)
            if "args_model" in ru.columns:
                am = dict(zip(ru.run_id.astype(str), ru.args_model.fillna("").astype(str)))
                rs["canon"] = [canon(am.get(str(r), "")) or canon(m) for r, m in zip(rs.run_id, rs.model)]
            else:
                rs["canon"] = [canon(m) for m in rs.model]
        elif "api_name" in rs.columns:                       # provider API model name (e.g. anthropic/claude-3.5-sonnet)
            rs["canon"] = [canon(a) or canon(m) for a, m in zip(rs.api_name.fillna("").astype(str), rs.model)]
        else:
            rs["canon"] = [canon(m) for m in rs.model]
        g = rs.groupby(["item_key", "model", "canon", "config"], dropna=False).outcome.agg(["mean", "size"]).reset_index()
        g["idx"] = g.item_key.map(keyidx); g["slug"] = d.name
        outs.append(g)
        print(f"{d.name}: {len(keyidx)} items, {rs.model.nunique()} raw models, mapped to canonical: {rs.canon.notna().mean():.0%}, "
              f"unmapped: {sorted(rs.model[rs.canon.isna()].unique())[:8]}", flush=True)
    LIB.mkdir(parents=True, exist_ok=True)
    h, i = np.asarray(sh_h, np.int64), np.asarray(sh_i, np.int32)
    o = np.argsort(h, kind="stable"); np.save(LIB / "sh_hash.npy", h[o]); np.save(LIB / "sh_item.npy", i[o])
    pd.DataFrame(items).to_parquet(LIB / "items.parquet")
    O = pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()
    O["year"] = O.canon.map(years); O.to_parquet(LIB / "outcomes.parquet")
    print(f"library: {len(items)} items from {len(outs)} benchmarks, {len(h)} shingles, {len(O)} (item, model, config) outcomes")


class Library:
    def __init__(self, path: Path = LIB):
        self.h = np.load(path / "sh_hash.npy"); self.i = np.load(path / "sh_item.npy")
        self.items = pd.read_parquet(path / "items.parquet")
        O = pd.read_parquet(path / "outcomes.parquet")
        self.by_item = {k: g for k, g in O.groupby("idx")}

    def match(self, content, min_votes: int = 3, min_frac: float = 0.5):
        q = query_shingles(content)
        if not len(q): return None
        lo = np.searchsorted(self.h, q, "left"); hi = np.searchsorted(self.h, q, "right")
        hits = np.concatenate([self.i[a:b] for a, b in zip(lo, hi) if b > a]) if (hi > lo).any() else np.array([], np.int32)
        if not len(hits): return None
        ids, votes = np.unique(hits, return_counts=True)
        j = int(np.argmax(votes)); idx, v = int(ids[j]), int(votes[j])
        if int((votes == v).sum()) > 1: return None              # ambiguous: several library items tie (shared text)
        need = min(min_votes, int(self.items.n_sh.iloc[idx]))
        if v < need or v / self.items.n_sh.iloc[idx] < min_frac: return None
        return idx, v / self.items.n_sh.iloc[idx]

    def outcomes(self, idx: int) -> pd.DataFrame:
        return self.by_item.get(idx, pd.DataFrame())


def selftest():
    """Recall on wrapped library texts, false positives on unrelated measurement-db items."""
    lib = Library(); rng = np.random.default_rng(0)
    rows = []
    for d in sorted(p for p in SRC.iterdir() if p.name in set(lib.items.slug)):
        it = pd.read_parquet(d / "items.parquet").dropna(subset=["text"])
        it = it[it.item_key.isin(set(lib.items.item_key[lib.items.slug == d.name]))]
        for r in it.sample(min(60, len(it)), random_state=0).itertuples():
            wrapped = json.dumps({"prompt": "You are an expert. Solve the following task carefully.\n\n" + r.text + "\n\nAnswer:", "meta": {"id": "x"}})
            m = lib.match(wrapped); ok = m is not None and lib.items.item_key.iloc[m[0]] == r.item_key and lib.items.slug.iloc[m[0]] == d.name
            rows.append(dict(slug=d.name, ok=ok, has_text=len(TOK.findall(str(r.text).lower())) >= 3))
    R = pd.DataFrame(rows); print("recall on wrapped library items:", R.groupby("slug").ok.mean().round(3).to_dict(), "| overall", round(R.ok.mean(), 3))
    fp, n = 0, 0
    for b in sorted((ROOT / "data" / "mdb_full").iterdir())[:60]:
        p = b / "formatted_tables" / "items.parquet"
        if not p.exists(): continue
        c = pd.read_parquet(p, columns=["content"]).content.dropna()
        for x in c.sample(min(30, len(c)), random_state=0): n += 1; fp += lib.match(x) is not None
    print(f"false positives on unrelated measurement-db items: {fp}/{n}")





def export(dest: str):
    """Write runtime artifacts (numpy + JSON only) for the submission: publib/{sh_hash.npy, sh_item.npy, items.json, outcomes.json}."""
    d = Path(dest); d.mkdir(parents=True, exist_ok=True)
    np.save(d / "sh_hash.npy", np.load(LIB / "sh_hash.npy")); np.save(d / "sh_item.npy", np.load(LIB / "sh_item.npy"))
    it = pd.read_parquet(LIB / "items.parquet")
    json.dump([[r.slug, int(r.n_sh), float(r.mean), int(r.n)] for r in it.itertuples()], open(d / "items.json", "w"))
    O = pd.read_parquet(LIB / "outcomes.parquet")
    out = {}
    for r in O.itertuples():
        out.setdefault(str(int(r.idx)), []).append([r.canon if isinstance(r.canon, str) else "", str(r.model), str(r.config if isinstance(r.config, str) else ""), float(r.mean), int(r.size)])
    json.dump(out, open(d / "outcomes.json", "w"))
    print(f"exported {len(it)} items, {len(O)} outcome rows to {d}")


if __name__ == "__main__":
    if sys.argv[1] == "export": export(sys.argv[2])
    else: {"build": build, "selftest": selftest}[sys.argv[1]]()
