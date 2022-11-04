import os
import launch

if not launch.is_installed("scikit_learn"):
    launch.run_pip("install scikit_learn", "requirements for Advance Prompt Tuning")


if not launch.is_installed("requests"):
    launch.run_pip("install requests", "requirements for Advance Prompt Tuning")


convnext_repo = os.environ.get('CONVNEXT_REPO', 'https://github.com/facebookresearch/ConvNeXt.git')
convnext_commit_hash = os.environ.get('CONVNEXT_COMMIT_HASH', "048efcea897d999aed302f2639b6270aedf8d4c8")

launch.git_clone(convnext_repo, launch.repo_dir('conv_next'), "conv_next", convnext_commit_hash)

# rename conv_next models to avoid conflict with BLIP
if os.path.exists(os.path.join(launch.repo_dir('conv_next'), 'models')):
    os.rename(os.path.join(launch.repo_dir('conv_next'), 'models'), os.path.join(launch.repo_dir('conv_next'), 'models_cnx'))