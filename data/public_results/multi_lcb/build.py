"""Build items.parquet / results.parquet / models.csv for Multi-LCB.

Usage: python -I build.py SITE CODE LCB_DIR OUT
  SITE    = checkout of https://github.com/Multi-LCB/multi-lcb.github.io at SITE_SHA (data/lcb/)
  CODE    = checkout of https://github.com/Multi-LCB/Multi-LCB at CODE_SHA (branch `release`)
  LCB_DIR = <file>.slim.jsonl of https://huggingface.co/datasets/livecodebench/code_generation_lite at
            revision REV (test.jsonl..test6.jsonl = release_v6), private test cases removed (see fetch.py)

Prompt text is rebuilt with the pinned Multi-LCB code. Two pure-stdlib modules
(prompt_enhancer/example_parser.py, prompt_enhancer/unified_converter.py) are loaded by file path.
The rest of the template is copied below and checked against the source.

Outcome = `pass` in data/lcb/<model>/<language>.jsonl: the fraction of the model's n samples
(meta.json `pass_at_n`) that passed every public and private test. One row per model x language x problem.
"""
import ast, collections, importlib.util, json, os, sys

SITE_SHA = "e1f940ea1a767d46ff6ef09411cafa3917524c8a"
CODE_SHA = "d80be9f24268e9e5c6e1463e26c9f05417fe06a0"
REV = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]

# site language key -> Multi-LCB plang (PLang value, lower case)
SITE2PLANG = {"cpp": "c++", "csharp": "c#", "python": "python", "java": "java", "rust": "rust", "go": "go",
              "typescript": "typescript", "javascript": "javascript", "ruby": "ruby", "php": "php",
              "scala": "scala", "kotlin": "kotlin"}
# copied from lcb_runner/utils/plangs.py (PLang.get_display_lang_name / get_markdown_lang_name / comment)
DISPLAY = {"c++": "C++", "c#": "C#", "javascript": "JavaScript", "typescript": "TypeScript", "python": "Python",
           "java": "Java", "rust": "Rust", "go": "Go", "ruby": "Ruby", "php": "php", "scala": "Scala", "kotlin": "Kotlin"}
MARKDOWN = {"c++": "cpp", "c#": "csharp", "javascript": "javascript", "typescript": "typescript", "python": "python",
            "java": "java", "rust": "rust", "go": "go", "ruby": "ruby", "php": "php", "scala": "scala", "kotlin": "kotlin"}
TWO_D_V2 = "For 2D arrays, the first line indicates the number of rows, followed by newline-separated rows."
CODE_FENCE = "```<param:plang>\n<param:comment> YOUR CODE HERE\n```\n\n"
ANSWER = "### Answer: (use the provided format with backticks)\n\n"

# Release month of each model as named on the site (public release of the weights / API). Sources: model cards
# and vendor announcements. "approx" = year certain, month not confirmed by a source.
RELEASE = {
    "DeepSeek-R1-0528": "2025-05", "DeepSeek-R1-Distill-Qwen-14B": "2025-01", "DeepSeek-R1-Distill-Qwen-32B": "2025-01",
    "DeepSeek-R1-Distill-Qwen-8B": "2025-05",  # data identical to the hidden dir _del_DeepSeek-R1-0528-Qwen3-8B_cot
    "Deepseek-Coder-33B-Instr": "2023-11", "Devstral-Small-2505": "2025-05", "GLM-4.5-Air": "2025-07",
    "GLM-4.7": "2025-12", "GLM-4.7-Flash": "2026-01", "GLM-5-FP8": "2026-02", "GLM-5.1-FP8": "2026-04 (approx)",
    "GPT-OSS-120B": "2025-08", "GPT-OSS-120B (High)": "2025-08", "GPT-OSS-20B": "2025-08",
    "GPT-OSS-20B (High)": "2025-08", "GPT-OSS-20B (Medium)": "2025-08", "MiniMax-M2.7": "2026-03 (approx)",
    "Nemotron-Cascade-14B-Thinking": "2025-12 (approx)", "O4-Mini": "2025-04", "OlympicCoder-32B": "2025-03",
    "OlympicCoder-7B": "2025-03", "OpenCodeRsn-Nmt-1.1-32B": "2025-07 (approx)", "OpenCoder-8B": "2024-11",
    "OpenRsn-Nmt-32B": "2025-07", "Ouro-1.4B-Thinking": "2025-10", "Ouro-2.6B-Thinking": "2025-10",
    "Qwen2.5-Coder-14B-Instr": "2024-11", "Qwen2.5-Coder-32B-Instr": "2024-11", "Qwen3-14B": "2025-04",
    "Qwen3-235B-A22B-Instr-2507": "2025-07", "Qwen3-235B-A22B-Thk-2507": "2025-07", "Qwen3-30B-A3B": "2025-04",
    "Qwen3-30B-A3B-Instr-2507": "2025-07", "Qwen3-30B-A3B-Thk-2507": "2025-07", "Qwen3-32B": "2025-04",
    "Qwen3-8B": "2025-04", "Qwen3-Coder-30B-A3B-Instr": "2025-07", "Qwen3-Coder-Next": "2026-02",
    "Qwen3.6-27B": "2026-04 (approx)", "Qwen3.6-35B-A3B": "2026-04 (approx)", "Seed-Coder-8B-Instr": "2025-05",
    "gemini-2.5-flash": "2025-06",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses needs the module registered
    spec.loader.exec_module(mod)
    return mod


def prompt_constants(code):
    """Read PromptConstants string literals from the pinned source without executing it."""
    src = open(f"{code}/lcb_runner/prompts/code_generation.py").read()
    consts = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "PromptConstants":
            for st in node.body:
                if isinstance(st, ast.Assign):
                    consts[st.targets[0].id] = ast.literal_eval(st.value)
    for frag in ['base_prompt = f"### Question:\\n{question.question_content}\\n\\n"',
                 'enhanced_prompt += "```<param:plang>\\n<param:comment> YOUR CODE HERE\\n```\\n\\n"',
                 'enhanced_prompt += "### Answer: (use the provided format with backticks)\\n\\n"',
                 "question.starter_code = None", "question.platform == Platform.LEETCODE"]:
        assert frag in src, frag
    enh = open(f"{code}/prompt_enhancer/enhancer.py").read()
    for frag in [TWO_D_V2, 'return f"{base_instructions} {two_d_instructions}"',
                 'enhanced_prompt = original_prompt + "\\n\\n"', "Sample Input {example['example_id']}:\n\n"]:
        assert frag in enh, frag
    parser_src = open(f"{code}/lcb_runner/runner/parser.py").read()
    assert 'os.environ.get("FORMAT_VERSION", "v2")' in parser_src
    assert 'os.environ.get("ENHANCE_PROMPTS", "True") == "True"' in parser_src
    assert 'os.environ.get("RELEASE_VERSION", "release_v6")' in parser_src
    return consts


class PromptBuilder:
    def __init__(self, code):
        c = prompt_constants(code)
        self.system = c["SYSTEM_MESSAGE_GENERIC"]
        self.fmt = c["FORMATTING_WITHOUT_STARTER_CODE"]
        ep = load_module("mlcb_example_parser", f"{code}/prompt_enhancer/example_parser.py")
        uc = load_module("mlcb_unified_converter", f"{code}/prompt_enhancer/unified_converter.py")
        self.parser = ep.ExampleParser()
        self.conv = uc.ConverterFactory.create_converter("v2")  # default FORMAT_VERSION at CODE_SHA

    def user_core(self, q):
        """User message before <param:...> substitution; mirrors get_enhanced_question_template_answer."""
        base = f"### Question:\n{q['question_content']}\n\n"
        if q["platform"] == "leetcode":
            out, variant, n_ex = base, "leetcode_no_examples_parsed", 0
            try:
                raw = self.parser.extract_examples(q["question_content"])
                proc = self.conv.convert_all_tests(raw)
                if raw:
                    out = base + "\n\n" + f"### Format: {self.fmt}\n" + " " + TWO_D_V2 + "\n\n"
                    for r, p in zip(raw, proc):
                        out += (f"Sample Input {r['example_id']}:\n\n{p['stdin']}\n\n"
                                f"Sample Output {r['example_id']}:\n\n{p['stdout']}\n\n") + "\n"
                    variant, n_ex = "leetcode_stdin_examples", len(raw)
            except Exception as e:  # enhancer falls back to the plain question on any error
                variant = f"leetcode_enhancer_error:{type(e).__name__}"
            return out + CODE_FENCE + ANSWER, variant, n_ex
        return base + f"### Format: {self.fmt}\n" + CODE_FENCE + ANSWER, "stdin_native", 0

    def render(self, q, site_lang, cache={}):
        plang = SITE2PLANG[site_lang]
        if q["question_id"] not in cache:
            cache[q["question_id"]] = self.user_core(q)
        core, variant, n_ex = cache[q["question_id"]]
        params = {"<param:Plang>": DISPLAY[plang], "<param:plang>": MARKDOWN[plang], "<param:PLANG>": plang.upper(),
                  "<param:comment>": "#" if plang in ("python", "ruby") else "//"}
        system, user = self.system, core
        for k, v in params.items():
            system, user = system.replace(k, v), user.replace(k, v)
        return f"{system}\n\n{user}", variant, n_ex


def build(site, code, lcb_dir, out):
    import pandas as pd
    lcb = {}
    for f in LCB_FILES:
        for line in open(f"{lcb_dir}/{f.replace('.jsonl', '.slim.jsonl')}"):
            q = json.loads(line)
            assert q["question_id"] not in lcb, q["question_id"]
            lcb[q["question_id"]] = q
    print("LCB release_v6 problems:", len(lcb))

    D = f"{site}/data/lcb"
    index = json.load(open(f"{D}/index.json"))
    live = [m for m in index["models"] if not m["name"].startswith(("_old", "_del"))]
    hidden = [m["name"] for m in index["models"] if m["name"].startswith(("_old", "_del"))]

    pb = PromptBuilder(code)
    res, items, models, mism = [], {}, [], collections.Counter()
    for m in live:
        name = m["name"]
        meta = json.load(open(f"{D}/{name}/meta.json"))
        n = int(meta["pass_at_n"])
        config = (f"temperature={meta['temperature']};top_p={meta['top_p']};"
                  f"reasoning={str(meta['reasoning']).lower()};n_samples={n}")
        probs, langs = set(), []
        for lang, fname in sorted(m["files"].items()):
            assert fname == f"{lang}.jsonl", (name, fname)
            langs.append(lang)
            for line in open(f"{D}/{name}/{fname}"):
                if not line.strip():
                    continue
                r = json.loads(line)
                tid, p = str(r["task_id"]), r["pass"]
                assert isinstance(p, (int, float)) and 0 <= p <= 1, (name, lang, r)
                assert abs(p * n - round(p * n)) < 1e-6, (name, lang, r, n)  # multiple of 1/n
                q = lcb[tid]  # KeyError = problem without a statement
                if r["date"] != q["contest_date"]:
                    mism["date"] += 1
                if "difficulty" in r and r["difficulty"] != q["difficulty"]:
                    mism["difficulty"] += 1
                if "platform" in r and r["platform"] != q["platform"]:
                    mism["platform"] += 1
                key = f"{lang}/{tid}"
                if key not in items:
                    text, variant, n_ex = pb.render(q, lang)
                    extra = {"language": lang, "language_display": DISPLAY[SITE2PLANG[lang]], "problem_id": tid,
                             "question_title": q["question_title"], "platform": q["platform"],
                             "difficulty": q["difficulty"], "contest_id": q["contest_id"],
                             "contest_date": q["contest_date"], "lcb_file": q["source_file"],
                             "prompt_variant": variant, "n_examples_converted": n_ex}
                    items[key] = {"item_key": key, "text": text, "extra": json.dumps(extra, ensure_ascii=False)}
                res.append({"item_key": key, "model": name, "config": config, "trial": 1, "outcome": float(p)})
                probs.add(tid)
        rel = RELEASE[name]
        models.append({"model": name, "config": config, "n_samples_per_problem": n,
                       "temperature": meta["temperature"], "top_p": meta["top_p"], "reasoning": meta["reasoning"],
                       "n_languages": len(langs), "languages": ",".join(langs), "n_problems": len(probs),
                       "n_results": sum(1 for x in res if x["model"] == name),
                       "release": rel, "release_2025_2026": rel[:4] in ("2025", "2026")})

    items_df = pd.DataFrame(list(items.values())).sort_values("item_key").reset_index(drop=True)
    res_df = pd.DataFrame(res).astype({"trial": "int64", "outcome": "float64"})
    assert items_df.item_key.is_unique
    assert not res_df.duplicated(["item_key", "model", "config", "trial"]).any()
    assert set(res_df.item_key) == set(items_df.item_key)
    items_df.to_parquet(f"{out}/items.parquet", index=False)
    res_df.to_parquet(f"{out}/results.parquet", index=False)
    mdf = pd.DataFrame(models).sort_values("model")
    mdf.to_csv(f"{out}/models.csv", index=False)

    ex = items_df.extra.map(json.loads)
    print("site/LCB metadata mismatches:", dict(mism))
    print("items", len(items_df), "results", len(res_df), "models", res_df.model.nunique(),
          "models 2025-2026", int(mdf.release_2025_2026.sum()))
    print("problems", ex.map(lambda e: e["problem_id"]).nunique(), "languages",
          sorted(ex.map(lambda e: e["language"]).unique()))
    print("prompt variants:", ex.map(lambda e: e["prompt_variant"]).value_counts().to_dict())
    per_prob = {e["problem_id"]: (e["platform"], e["difficulty"], e["contest_date"][:7]) for e in ex}
    print("problems by platform:", collections.Counter(v[0] for v in per_prob.values()))
    print("problems by difficulty:", collections.Counter(v[1] for v in per_prob.values()))
    print("contest months:", min(v[2] for v in per_prob.values()), "..", max(v[2] for v in per_prob.values()))
    print("items per language:", collections.Counter(e["language"] for e in ex))
    print("hidden dirs skipped:", len(hidden))
    print(mdf[["model", "n_samples_per_problem", "n_languages", "n_problems", "n_results", "release"]].to_string())


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    build(*sys.argv[1:5])
