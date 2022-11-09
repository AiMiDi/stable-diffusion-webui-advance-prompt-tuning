import os
from launch import git, run, run_pip, is_installed
from pathlib import Path

REPO_LOCATION = Path(__file__).parent
auto_update = bool(os.environ.get("AUTO_UPDATE", "True"))

if auto_update:
    print("[advance-prompt-tuning] Attempting auto-update...")

    try:
        run(f'"{git}" -C {REPO_LOCATION} fetch', "[advance-prompt-tuning] Fetch upstream.")
        run(f'"{git}" -C {REPO_LOCATION} pull', "[advance-prompt-tuning] Pull upstream.")
    except Exception as e:
        print("[advance-prompt-tuning] Auto-update failed:")
        print(e)
        print("[advance-prompt-tuning] Ensure git was used to install extension.")

if not is_installed("scikit_learn"):
    run_pip("install scikit_learn", "requirements for Advance Prompt Tuning")

if not is_installed("requests"):
    run_pip("install requests", "requirements for Advance Prompt Tuning")

if not is_installed("tqdm"):
    run_pip("install tqdm", "requirements for Advance Prompt Tuning")