import os
import sys
import launch
from launch import git, run
from pathlib import Path
from modules.paths import sd_path, paths

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

if not launch.is_installed("scikit_learn"):
    launch.run_pip("install scikit_learn", "requirements for Advance Prompt Tuning")

if not launch.is_installed("requests"):
    launch.run_pip("install requests", "requirements for Advance Prompt Tuning")

if not launch.is_installed("tqdm"):
    launch.run_pip("install tqdm", "requirements for Advance Prompt Tuning")

convnext_repo = os.environ.get('CONVNEXT_REPO', 'https://github.com/facebookresearch/ConvNeXt.git')
convnext_commit_hash = os.environ.get('CONVNEXT_COMMIT_HASH', "048efcea897d999aed302f2639b6270aedf8d4c8")

launch.git_clone(convnext_repo, launch.repo_dir('conv_next'), "conv_next", convnext_commit_hash)

# rename conv_next models to avoid conflict with BLIP
if os.path.exists(os.path.join(launch.repo_dir('conv_next'), 'models')):
    os.rename(os.path.join(launch.repo_dir('conv_next'), 'models'), os.path.join(launch.repo_dir('conv_next'), 'models_cnx'))

model_path = os.path.join(sd_path, '../conv_next')
must_exist_path = os.path.abspath(os.path.join(model_path, 'models_cnx/convnext.py'))
if not os.path.exists(must_exist_path):
    print(f"Warning: conv_next not found at path {must_exist_path}")
else:
    model_path = os.path.abspath(model_path)
    print(f"add model path: [conv_next] - {model_path}")
    sys.path.append(model_path)
    paths['conv_next'] = model_path