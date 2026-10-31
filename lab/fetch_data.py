"""Download the training data this project uses, at pinned revisions (run from the project root).

  python lab/fetch_data.py core     # the six competition benchmarks -> data/measurement-db/ (rev bc8204d8, ~65 MB)
  python lab/fetch_data.py core --check   # only verify that access works (downloads one small file)
  python lab/fetch_data.py full     # item tables of 59 benchmarks of the larger checkpoint -> data/mdb_full/
                                    # (rev 54074453, ~425 MB; used only for the false-match check of h22_library.py selftest)

measurement-db is gated: request access once at https://huggingface.co/datasets/aims-foundations/measurement-db
and log in with `hf auth login` (or put a token in ~/.cache/huggingface/token).
"""
import sys
from huggingface_hub import hf_hub_download, snapshot_download

REPO = "aims-foundations/measurement-db"
CORE_REV = "bc8204d811823da849c6686bf124d4ca9f82e4de"   # "Archive reproducibility pilots for six public benchmarks"
FULL_REV = "54074453da7b9911be0f48e4c560be547bc84d21"   # "Move six legacy benchmark releases into formatted_tables"
CORE = ["matharena", "mmdocrag", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]
# the selftest samples 30 items from each of the first 60 entries of data/mdb_full in sorted order (.cache + these 59)
SELFTEST = ["adaptivestep", "afrieval", "afrimedqa", "agc_bench", "agentdojo", "agentic_review_perturbation",
            "agi_elo", "ai2d_test", "algotune", "alignment_faking", "alpacaeval", "annotating_errors_wcf",
            "appworld", "arcagi", "atmossci_bench", "babilong", "bbq", "beavertails", "bedd_basalt", "benger",
            "bertaqa", "bigfinancebench", "biggen", "bird_sql", "brace", "bridging_gap", "care_enzymes", "ceval",
            "chatgpt_drift", "chi_bench", "chipbench", "clasheval", "classroom_ai", "cmmlu", "complexbench",
            "corebench", "critic_discernment_game", "crow", "cruxeval", "csedb", "das_med_hallucination",
            "dataclawbench", "dave", "decodingtrust", "devbench", "disco", "donotanswer", "dpai", "editbench",
            "edu_circuit_hw", "eduguardbench", "egoschema", "elicitation_game", "embodied_agent_interface",
            "emoji_attack", "engdesign", "evouna", "faithcot", "fetv"]

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "core"
    if "--check" in sys.argv:
        p = hf_hub_download(REPO, "README.md", repo_type="dataset", revision=CORE_REV, local_dir="data/measurement-db")
        print("access OK:", p)
    elif what == "core":
        snapshot_download(REPO, repo_type="dataset", revision=CORE_REV, local_dir="data/measurement-db",
                          allow_patterns=[f"{b}/*" for b in CORE] + ["*.md", "*.yaml", "*.json"])
        print("core benchmarks in data/measurement-db")
    elif what == "full":
        snapshot_download(REPO, repo_type="dataset", revision=FULL_REV, local_dir="data/mdb_full",
                          allow_patterns=[f"{b}/formatted_tables/items.parquet" for b in SELFTEST])
        print("selftest item tables in data/mdb_full")
    else:
        raise SystemExit(__doc__)
