import os
import sys
import launch
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

convnext_repo = os.environ.get('CONVNEXT_REPO', 'https://github.com/facebookresearch/ConvNeXt.git')
convnext_commit_hash = os.environ.get('CONVNEXT_COMMIT_HASH', "048efcea897d999aed302f2639b6270aedf8d4c8")

launch.git_clone(convnext_repo, launch.repo_dir('ConvNeXt'), "ConvNeXt", convnext_commit_hash)

# rename ConvNeXt models to avoid conflict with BLIP
if os.path.exists(os.path.join(launch.repo_dir('ConvNeXt'), 'models')):
    os.rename(os.path.join(launch.repo_dir('ConvNeXt'), 'models'), os.path.join(launch.repo_dir('ConvNeXt'), 'ConvNeXt'))



