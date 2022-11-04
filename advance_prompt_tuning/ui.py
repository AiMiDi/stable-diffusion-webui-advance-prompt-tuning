import html

import gradio as gr

import advance_prompt_tuning.advance_prompt_tuning as apt
import advance_prompt_tuning.preprocess as apt_preprocess
from modules import sd_hijack, shared


def create_embedding(name, initialization_text, nvpt):
    filename = apt.create_embedding(name, nvpt, init_text=initialization_text)

    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

    return gr.Dropdown.update(choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())), f"Created: {filename}", ""


def preprocess(*args):
    apt_preprocess.preprocess(*args)

    return "Preprocessing finished.", ""


def train_embedding(*args):

    try:
        sd_hijack.undo_optimizations()

        embedding, filename = apt.train_embedding(*args)

        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'} at {embedding.step} steps.
Embedding saved to {html.escape(filename)}
"""
        return res, ""
    except Exception:
        raise
    finally:
        sd_hijack.apply_optimizations()

