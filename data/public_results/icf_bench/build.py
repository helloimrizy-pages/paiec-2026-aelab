"""Build items.parquet / results.parquet / judgments_raw.parquet / models.csv for ICF-Bench (ICLR 2026).

Source: https://github.com/qianyuli123/ICF-Bench at commit ac8149951516cf5770d78edaacc3435bb089417c.

Usage: python -I build.py <repo> <out_dir>   (fetch.py downloads <repo> at the commit and runs this)

Prompts are rebuilt exactly as the published answer.py scripts build them (message lists), then rendered as text.
Outcomes come only from the published judge_<model>.json files, using the pass rule of each scenario's calculate.py
(the script that produced the numbers in <scenario>/cal.txt).
"""
import json
import sys
from pathlib import Path

import pandas as pd

COMMIT = "ac8149951516cf5770d78edaacc3435bb089417c"

MODELS = {
    # release string -> (canonical model, release date, 2025-2026 flag)
    "gpt4o1120-Eval": ("GPT-4o (gpt-4o-2024-11-20)", "2024-11-20", False),
    "gpt-5-chat-shortco-2025-08-07-Eval": ("GPT-5 chat (gpt-5-chat, 2025-08-07)", "2025-08-07", True),
    "gpt-5-chat-shortco-2025-08-07-Bing": ("GPT-5 chat (gpt-5-chat, 2025-08-07)", "2025-08-07", True),
    "o3Mini-Eval": ("o3-mini (snapshot not stated, 2025-01-31 release)", "2025-01-31", True),
}
# judge model named in each scenario's published judge.py (which header set call_llm/call_model actually uses)
JUDGE = {
    "dynamic_preference": "gpt4o1120-Eval",
    "instructional_forgetting": "o4-mini-2025-04-16-Eval",
    "subtask_revision": "gpt4o1120-Eval",
}


def render(messages):
    """Render a chat message list as plain text: '[role]\\ncontent' blocks separated by a blank line."""
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)


# ---------------------------------------------------------------- dynamic_preference
DP_SYS = "You are a useful assistant. You can make the most reasonable choice according to my preferences"
DP_ACK = "I know your preference and will remember it."
LABELS = ["A.", "B.", "C.", "D."]


def dp_question(new_entry, only):
    option_text = "\n".join(f"{label} {text}" for label, text in zip(LABELS, new_entry["options"]))
    w = "you can only choose one" if only else "you can choose one"
    # exact template of answer.py (note the space after the options and the 4 trailing spaces)
    return (new_entry["question"]
            + f" Here are four options, {w} as your answer, just return the content of option, "
            + "no more additional descriptions or omissions.\n"
            + option_text + " \n    ")


def dp_messages(old_pref, insert_conv, new_entry, setting):
    msgs = [{"role": "system", "content": DP_SYS}]
    msgs += [{"role": "user", "content": old_pref}, {"role": "assistant", "content": DP_ACK}]
    if setting == "forget":  # answer.py "exist_old" branch
        msgs += [
            {"role": "user", "content": new_entry["conversation_query"]},
            {"role": "assistant", "content": new_entry["conversation_assistant_options"]},
            {"role": "user", "content": new_entry["conversation_user_selection"]},
            {"role": "assistant", "content": new_entry["conversation_assistant_acknowledgment"]},
        ]
        msgs += insert_conv
        msgs.append({"role": "user", "content": dp_question(new_entry, only=False)})
    else:  # "noexist_old" branch
        msgs += insert_conv
        msgs.append({"role": "user", "content": dp_question(new_entry, only=True)})
    return msgs


def build_dynamic_preference(repo, items, raw):
    d = repo / "dynamic_preference"
    old = {o["from_implicit_id"]: o for o in json.load(open(d / "dynamic_preference_old.json", encoding="utf-8"))}
    new = json.load(open(d / "dynamic_preference_new.json", encoding="utf-8"))
    for f in sorted(d.glob("judge_*.json")):
        model = f.stem[len("judge_"):]
        for e in json.load(open(f, encoding="utf-8")):
            i = int(e["id"])
            n = new[i]
            assert e["new_preference"] == n["preference"] and e["new_op"] == n["aligned_op"]
            assert e["question"] == dp_question(n, only=False)
            in_old = i in old
            if in_old:
                assert old[i]["preference"] == e["old_preference"] and old[i]["source_option"] == e["old_op"]
            insert_conv = old[i]["insert_conv"] if in_old else []
            for setting, resp_key, grade_key in [
                ("forget", "llm_response_exist_old", "match_old_exist_old"),
                ("noforget", "llm_response_noexist_old", "match_old_noexist_old"),
            ]:
                key = f"dynamic_preference/{i}/{setting}"
                if key not in items:
                    msgs = dp_messages(e["old_preference"], insert_conv, n, setting)
                    items[key] = dict(
                        text=render(msgs),
                        extra=dict(
                            scenario="dynamic_preference", id=i, setting=setting,
                            interference=(setting == "forget"),
                            last_user_turn=msgs[-1]["content"],
                            target_option=e["old_op"], conflicting_option=e["new_op"],
                            target_preference=e["old_preference"], conflicting_preference=e["new_preference"],
                            options=n["options"], topic=n.get("topic"),
                            in_current_old_json=in_old,
                            insert_conv_turns=len(insert_conv),
                        ),
                    )
                resp = e[resp_key]
                g = e[grade_key]
                api_fail = resp.strip() in ("", "ERROR")
                keep = isinstance(g, bool) and not api_fail
                raw.append(dict(
                    item_key=key, model=model, response=resp,
                    match_old=e[grade_key], match_new=e[grade_key.replace("match_old", "match_new")],
                    stored_success_flag=e["success_forget" if setting == "forget" else "success_noforget"],
                    judge_text="", api_failure=api_fail, kept=keep,
                    outcome=float(g) if keep else None,
                ))


# ---------------------------------------------------------------- instructional_forgetting
IF_SYS = "You are a helpful assistant. You should answer the last question based on history conversation."
IF_ACK = "Ok, I will follow your instructions and never mention them again."


def if_messages(entry, setting):
    msgs = [{"role": "system", "content": IF_SYS}]
    for conv in entry["conversations"][:-2]:
        role = "user" if conv["from"] == "human" else "assistant" if conv["from"] == "gpt" else conv["from"]
        msgs.append({"role": role, "content": conv["value"]})
    if setting == "forget":
        msgs.append({"role": "user", "content": entry["forget_instruction"]})
        msgs.append({"role": "assistant", "content": IF_ACK})
    msgs += entry.get("insert_conv", [])
    msgs.append({"role": "user", "content": entry["test_query"]})
    return msgs


def build_instructional_forgetting(repo, items, raw):
    d = repo / "instructional_forgetting"
    for f in sorted(d.glob("judge_*.json")):
        model = f.stem[len("judge_"):]
        for e in json.load(open(f, encoding="utf-8")):
            i = int(e["id"])
            for setting, resp_key, judge_key, flag_key in [
                ("forget", "instruction_forget_reply", "llm_judge_result_forget", "success_forget"),
                ("noforget", "instruction_noforget_reply", "llm_judge_result_noforget", "success_noforget"),
            ]:
                key = f"instructional_forgetting/{i}/{setting}"
                msgs = if_messages(e, setting)
                text = render(msgs)
                if key not in items:
                    items[key] = dict(
                        text=text,
                        extra=dict(
                            scenario="instructional_forgetting", id=i, setting=setting,
                            interference=(setting == "forget"),
                            last_user_turn=msgs[-1]["content"],
                            forget_instruction=e["forget_instruction"],
                            original_pair_index=e.get("original_pair_index"),
                            n_history_turns=len(e["conversations"]) - 2,
                        ),
                    )
                else:
                    assert items[key]["text"] == text
                resp = e[resp_key]
                flag = e[flag_key]
                jt = e[judge_key]
                assert (flag == "True") == ("yes" in jt.strip().lower())
                api_fail = resp.strip() == ""
                keep = flag in ("True", "False") and not api_fail and not jt.startswith("Error")
                raw.append(dict(
                    item_key=key, model=model, response=resp, match_old=None, match_new=None,
                    stored_success_flag=flag, judge_text=jt, api_failure=api_fail, kept=keep,
                    outcome=float(flag == "True") if keep else None,
                ))


# ---------------------------------------------------------------- subtask_revision
SR_SYS = "You are a usful assiatant, you can give the answer following my instruction."
OWN_ANSWER = "{MODEL_OWN_ANSWER_TO_ORIGINAL_INSTRUCTION}"


def sr_prompt(instruction):
    # exact template of subtask_revision/answer.py build_generation_prompt
    return ("You are a useful assistant. Based on the history conversation, you should give me a answer for my instruction.\n"
            + "        Instruction:\n"
            + f"            {instruction}\n"
            + "        ")


def sr_messages(entry, setting):
    msgs = [{"role": "system", "content": SR_SYS}]
    if setting == "forget":  # multi-turn: original instruction, model's own answer, then modified instruction
        msgs.append({"role": "user", "content": sr_prompt(entry["old_instruction"])})
        msgs.append({"role": "assistant", "content": OWN_ANSWER})
    msgs += entry.get("insert_conv", [])
    msgs.append({"role": "user", "content": sr_prompt(entry["new_instruction"])})
    return msgs


def sr_pass(j):
    """calculate.py rule: same_subtask_Judge True, modified_subtask_Judge True, original_subtask_Judge False."""
    return j.get("same_subtask_Judge") is True and j.get("modified_subtask_Judge") is True and j.get("original_subtask_Judge") is False


def sr_explicit(j):
    return isinstance(j, dict) and all(isinstance(j.get(k), bool) for k in ("same_subtask_Judge", "original_subtask_Judge", "modified_subtask_Judge"))


def build_subtask_revision(repo, items, raw):
    d = repo / "subtask_revision"
    for f in sorted(d.glob("judge_*.json")):
        model = f.stem[len("judge_"):]
        for pos, e in enumerate(json.load(open(f, encoding="utf-8"))):
            i = int(e["id"])
            assert i == pos
            for setting, judge_key in [("forget", "multi_judge"), ("noforget", "context_judge")]:
                key = f"subtask_revision/{i}/{setting}"
                msgs = sr_messages(e, setting)
                text = render(msgs)
                if key not in items:
                    items[key] = dict(
                        text=text,
                        extra=dict(
                            scenario="subtask_revision", id=i, setting=setting,
                            interference=(setting == "forget"),
                            last_user_turn=msgs[-1]["content"],
                            old_instruction=e["old_instruction"], new_instruction=e["new_instruction"],
                            common_subtasks=e["common_subtasks"],
                            original_special_subtasks=e["original_special_subtasks"],
                            modified_special_subtasks=e["modified_special_subtasks"],
                            text_has_model_answer_placeholder=(setting == "forget"),
                            unused_conversation_field_turns=len(e.get("conversation") or []),
                        ),
                    )
                else:
                    assert items[key]["text"] == text
                j = e[judge_key]
                if setting == "forget":
                    resp = json.dumps({"original": e["multi_original_answer"], "modified": e["multi_modified_answer"]}, ensure_ascii=False)
                    api_fail = (not e["multi_original_answer"].strip()) or (not e["multi_modified_answer"].strip())
                else:
                    resp = e["context_modified_answer"]
                    api_fail = not resp.strip()
                keep = sr_explicit(j) and not api_fail
                raw.append(dict(
                    item_key=key, model=model, response=resp, match_old=None, match_new=None,
                    stored_success_flag=e["success_forget" if setting == "forget" else "success_noforget"],
                    judge_text=json.dumps(j, ensure_ascii=False), api_failure=api_fail, kept=keep,
                    outcome=float(sr_pass(j)) if keep else None,
                ))


def main(repo, out):
    repo, out = Path(repo), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    items, raw = {}, []
    build_dynamic_preference(repo, items, raw)
    build_instructional_forgetting(repo, items, raw)
    build_subtask_revision(repo, items, raw)

    it = pd.DataFrame(
        [dict(item_key=k, text=v["text"], extra=json.dumps(v["extra"], ensure_ascii=False)) for k, v in sorted(items.items())]
    ).astype({"item_key": "string", "text": "string", "extra": "string"})
    assert it.item_key.is_unique

    rw = pd.DataFrame(raw)
    rw["scenario"] = rw.item_key.str.split("/").str[0]
    rw["setting"] = rw.item_key.str.split("/").str[2]
    rw["judge_model_per_code"] = rw.scenario.map(JUDGE)
    rw["source_file"] = rw.scenario + "/judge_" + rw.model + ".json"
    rw["release_commit"] = COMMIT
    for c in ["item_key", "model", "response", "stored_success_flag", "judge_text", "scenario", "setting",
              "judge_model_per_code", "source_file", "release_commit"]:
        rw[c] = rw[c].astype("string")
    for c in ["match_old", "match_new"]:
        rw[c] = rw[c].astype("boolean")

    res = rw[rw.kept].copy()
    res["config"] = ""
    res["trial"] = 1
    res = res[["item_key", "model", "config", "trial", "outcome"]].astype(
        {"item_key": "string", "model": "string", "config": "string", "trial": "int64", "outcome": "float64"}
    ).sort_values(["item_key", "model"]).reset_index(drop=True)
    assert not res.duplicated(["item_key", "model", "config", "trial"]).any()
    assert set(res.item_key) <= set(it.item_key)

    it.to_parquet(out / "items.parquet", index=False)
    res.to_parquet(out / "results.parquet", index=False)
    rw.to_parquet(out / "judgments_raw.parquet", index=False)

    rows = []
    for m, (canon, rel, recent) in MODELS.items():
        sub = res[res.model == m]
        rows.append(dict(
            model=m, canonical=canon, release_date=rel, release_2025_2026=recent,
            scenarios=",".join(sorted(sub.item_key.str.split("/").str[0].unique())),
            n_results=len(sub), n_dropped_api_or_parse=int(((rw.model == m) & ~rw.kept).sum()),
        ))
    pd.DataFrame(rows).to_csv(out / "models.csv", index=False)

    # summary
    print("items", len(it), "results", len(res), "raw", len(rw), "dropped", int((~rw.kept).sum()))
    print(it.item_key.str.split("/").str[0].value_counts().to_dict())
    print("duplicate texts:", int(it.text.duplicated(keep=False).sum()))
    g = rw.groupby(["scenario", "model", "setting"])
    summ = pd.DataFrame({"n_raw": g.size(), "n_kept": g.kept.sum()})

    # benchmark rate over all rows (dropped rows counted with the benchmark's own grade), to compare with cal.txt
    def bench_rate(x):
        sc = x.name[0]
        if sc == "dynamic_preference":
            return x.match_old.astype(float).mean()
        if sc == "instructional_forgetting":
            return (x.stored_success_flag == "True").mean()
        return x.judge_text.map(lambda t: sr_pass(json.loads(t))).mean()
    summ["rate_all_rows_benchmark_rule"] = g.apply(bench_rate)
    summ["rate_kept_rows"] = g.outcome.mean()
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
