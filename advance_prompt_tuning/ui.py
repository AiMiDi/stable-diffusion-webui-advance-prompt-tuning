import html

import gradio as gr

import advance_prompt_tuning.advance_prompt_tuning as apt
from modules import sd_hijack, shared


def create_advance_prompt_tuning_embedding(name, nvpt, overwrite_old, initialization_text, nvpt_uc, use_negative):
    filename = apt.create_apt_embedding(name, nvpt, overwrite_old, use_negative, init_text=initialization_text)
    if use_negative:
        apt.create_apt_embedding(name+'-uc', nvpt_uc, overwrite_old, use_negative, init_text=initialization_text)
        filename=f'{filename} and {filename[:-3]}-uc.pt'

    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

    return gr.Dropdown.update(choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())),f"Created: {filename}", ""


def train_advance_prompt_tuning_embedding(*args):

    try:
        sd_hijack.undo_optimizations()

        embedding, filename = apt.train_apt_embedding(*args)

        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'} at {embedding.step} steps.
Embedding saved to {html.escape(filename)}
"""
        return res, ""
    except Exception:
        raise
    finally:
        sd_hijack.apply_optimizations()

