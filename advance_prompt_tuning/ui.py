import html

import gradio as gr

import advance_prompt_tuning.advance_prompt_tuning as apt
from modules import sd_hijack, shared


def create_advance_prompt_tuning_embedding(name, nvpt, nvucpt, overwrite_old, initialization_text):
    filename, uc_filename = apt.create_embedding(name, nvpt, nvucpt, overwrite_old, initialization_text)

    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
    
    return gr.Dropdown.update(choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())), f"Created: embedding: {filename} and uc_embedding: {uc_filename}", ""


def train_advance_prompt_tuning_embedding(*args):

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

