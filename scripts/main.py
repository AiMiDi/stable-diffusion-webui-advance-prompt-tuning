import os
import gradio as gr

import advance_prompt_tuning.ui as apt_ui
from modules.sd_hijack import model_hijack
from modules import shared, script_callbacks
from modules.script_callbacks import UiTrainTabParams
from modules.paths import script_path
from modules.ui import create_refresh_button, setup_progressbar
from webui import wrap_gradio_gpu_call


def create_tabs(params:UiTrainTabParams):
    with gr.Tab(label="Advance prompt tuning") as advance_prompt_tuning_tab:
        with gr.Row().style(equal_height=False):
            gr.HTML(value="<p style='margin-bottom: 0.7em'>See <b><a href=\"=https://github.com/AiMiDi/stable-diffusion-webui-advance-prompt-tuning\">wiki</a></b> for detailed explanation.</p>")
        with gr.Row().style(equal_height=False):
            with gr.Tabs():
                with gr.Tab(label="Create advance prompt tuning embedding"):
                    apt_new_embedding_name = gr.Textbox(label="Name")
                    apt_initialization_text = gr.Textbox(
                        label="Initialization text", value="*")
                    apt_nvpt = gr.Slider(
                        label="Number of vectors per token", minimum=1, maximum=75, step=1, value=3)
                    apt_nvucpt = gr.Slider(
                        label="Number of negative vectors per token", minimum=1, maximum=75, step=1, value=10)
                    apt_overwrite_old_embedding = gr.Checkbox(
                        value=False, label="Overwrite Old Embedding")

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            create_apt_embedding = gr.Button(
                                value="Create embedding", variant='primary')

                with gr.Tab(label="Advance prompt tuning train"):
                    with gr.Row():
                        train_apt_embedding_name = gr.Dropdown(label='Embedding', choices=sorted(
                            model_hijack.embedding_db.word_embeddings.keys()))
                        create_refresh_button(train_apt_embedding_name, model_hijack.embedding_db.load_textual_inversion_embeddings, lambda: {
                                              "choices": sorted(model_hijack.embedding_db.word_embeddings.keys())}, "refresh_train_embedding_name")
                    with gr.Row():
                        embedding_learn_rate = gr.Textbox(
                            label='Embedding Learning rate', placeholder="Embedding Learning rate", value="0.005")
                        negative_scale_rate = gr.Textbox(
                            label='Negtive scale', placeholder="APT Negtive scale", value="5")

                    batch_size = gr.Number(
                        label='Batch size', value=1, precision=0)
                    dataset_directory = gr.Textbox(
                        label='Dataset directory', placeholder="Path to directory with input images")
                    log_directory = gr.Textbox(
                        label='Log directory', placeholder="Path to directory where to write outputs", value="textual_inversion")
                    disc_path = gr.Textbox(
                        label='ConvNext model path', placeholder="Path to ConvNext model ckpt", value="models/convnext/checkpoint-best_t5.pth")
                    template_file = gr.Textbox(label='Prompt template file', value=os.path.join(
                        script_path, "textual_inversion_templates", "style_filewords.txt"))
                    training_width = gr.Slider(
                        minimum=64, maximum=2048, step=64, label="Width", value=512)
                    training_height = gr.Slider(
                        minimum=64, maximum=2048, step=64, label="Height", value=512)
                    steps = gr.Number(label='Max steps',
                                      value=100000, precision=0)
                    create_image_every = gr.Number(
                        label='Save an image to log directory every N steps, 0 to disable', value=500, precision=0)
                    save_embedding_every = gr.Number(
                        label='Save a copy of embedding to log directory every N steps, 0 to disable', value=500, precision=0)
                    save_image_with_stored_embedding = gr.Checkbox(
                        label='Save images with embedding in PNG chunks', value=True)
                    preview_from_txt2img = gr.Checkbox(
                        label='Read parameters (prompt, etc...) from txt2img tab when making previews', value=False)

                    with gr.Row():
                        interrupt_training = gr.Button(value="Interrupt")
                        train_apt_embedding = gr.Button(
                            value="Train APT Embedding", variant='primary')
        apt_output = gr.Text(elem_id="apt_output", value="", show_label=False)
        create_apt_embedding.click(
            fn=apt_ui.create_advance_prompt_tuning_embedding,
            #_js="create_apt_embedding",
            inputs=[
                apt_new_embedding_name,
                apt_nvpt,
                apt_nvucpt,
                apt_overwrite_old_embedding,
                apt_initialization_text
            ],
            outputs=[
                train_apt_embedding_name,
                apt_output
            ]
        )

        train_apt_embedding.click(
            fn=wrap_gradio_gpu_call(apt_ui.train_advance_prompt_tuning_embedding, extra_outputs=[gr.update()]),
            #_js="start_apt_tuning",
            inputs=[
                train_apt_embedding_name,
                embedding_learn_rate,
                negative_scale_rate,
                dataset_directory,
                template_file,
                steps,
                save_embedding_every,
                create_image_every,
                disc_path,
                log_directory,
                training_width,
                training_height,
                save_image_with_stored_embedding,
                preview_from_txt2img,
                *params.txt2img_preview_params
            ],
            outputs=[
                apt_output
            ]
        )

        interrupt_training.click(
            fn=lambda: shared.state.interrupt(),
            inputs=[],
            outputs=[],
        )

    return advance_prompt_tuning_tab

script_callbacks.on_ui_train_tabs(create_tabs)
