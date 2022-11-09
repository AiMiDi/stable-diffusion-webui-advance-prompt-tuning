import os
import sys
import traceback

import torch
import tqdm
import html
import datetime
from torch.nn import functional as torch_functional

from PIL import PngImagePlugin
from modules import shared, devices, sd_hijack, processing, sd_models, images
import advance_prompt_tuning.dataset as apt_dataset
from conv_next.interface import XPDiscriminator
from modules.textual_inversion.textual_inversion import write_loss
from modules.textual_inversion.image_embedding import embedding_to_b64, insert_image_data_embed, caption_image_overlay

class AdvancePromptTuning:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None

    def save(self, filename):
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }

        torch.save(embedding_data, filename)

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        self.cached_checksum = f'{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}'
        return self.cached_checksum


class AdvancePromptTuningDatabase:
    def __init__(self, embeddings_dir):
        self.ids_lookup = {}
        self.word_embeddings = {}
        self.dir_mtime = None
        self.embeddings_dir = embeddings_dir

    def register_embedding(self, embedding, model):

        self.word_embeddings[embedding.name] = embedding

        ids = model.cond_stage_model.tokenizer([embedding.name], add_special_tokens=False)['input_ids'][0]

        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []

        self.ids_lookup[first_id] = sorted(self.ids_lookup[first_id] + [(ids, embedding)], key=lambda x: len(x[0]), reverse=True)

        return embedding

    def load_textual_inversion_embeddings(self):
        mt = os.path.getmtime(self.embeddings_dir)
        if self.dir_mtime is not None and mt <= self.dir_mtime:
            return

        self.dir_mtime = mt
        self.ids_lookup.clear()
        self.word_embeddings.clear()

        def process_file(path, filename):
            name = os.path.splitext(filename)[0]

            data = torch.load(path, map_location="cpu")

            # textual inversion embeddings
            if 'string_to_param' in data:
                param_dict = data['string_to_param']
                if hasattr(param_dict, '_parameters'):
                    param_dict = getattr(param_dict, '_parameters')  # fix for torch 1.12.1 loading saved file from torch 1.11
                assert len(param_dict) == 1, 'embedding file has multiple terms in it'
                emb = next(iter(param_dict.items()))[1]
            # diffuser concepts
            elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
                assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

                emb = next(iter(data.values()))
                if len(emb.shape) == 1:
                    emb = emb.unsqueeze(0)
            else:
                raise Exception(f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept.")

            vec = emb.detach().to(devices.device, dtype=torch.float32)
            embedding = AdvancePromptTuning(vec, name)
            embedding.step = data.get('step', None)
            embedding.sd_checkpoint = data.get('hash', None)
            embedding.sd_checkpoint_name = data.get('sd_checkpoint_name', None)
            self.register_embedding(embedding, shared.sd_model)

        for fn in os.listdir(self.embeddings_dir):
            try:
                fullfn = os.path.join(self.embeddings_dir, fn)

                if os.stat(fullfn).st_size == 0:
                    continue

                process_file(fullfn, fn)
            except Exception:
                print(f"Error loading emedding {fn}:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                continue

        print(f"Loaded a total of {len(self.word_embeddings)} textual inversion embeddings.")

    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None


def create_embedding(name, num_vectors_per_token, num_vectors_uc_per_token, overwrite_old, init_text='*'):
    cond_model = shared.sd_model.cond_stage_model
    embedding_layer = cond_model.wrapped.transformer.text_model.embeddings

    ids = cond_model.tokenizer(init_text, max_length=num_vectors_per_token, return_tensors="pt", add_special_tokens=False)["input_ids"]
    embedded = embedding_layer.token_embedding.wrapped(ids.to(devices.device)).squeeze(0)
    vec = torch.zeros((num_vectors_per_token, embedded.shape[1]), device=devices.device)

    for i in range(num_vectors_per_token):
        vec[i] = embedded[i * int(embedded.shape[0]) // num_vectors_per_token]

    uc_ids = cond_model.tokenizer(init_text, max_length=num_vectors_uc_per_token, return_tensors="pt", add_special_tokens=False)["input_ids"]
    uc_embedded = embedding_layer.token_embedding.wrapped(uc_ids.to(devices.device)).squeeze(0)
    uc_vec = torch.zeros((num_vectors_uc_per_token, embedded.shape[1]), device=devices.device)

    for i in range(num_vectors_uc_per_token):
        uc_vec[i] = uc_embedded[i * int(uc_embedded.shape[0]) // num_vectors_uc_per_token]

    # Remove illegal characters from name.
    name = "".join( x for x in name if (x.isalnum() or x in "._- "))
    uc_name = name+'-uc'

    fn = os.path.join(shared.cmd_opts.embeddings_dir, f"{name}.pt")
    uc_fn = os.path.join(shared.cmd_opts.embeddings_dir, f"{uc_name}.pt")
    if not overwrite_old:
        assert not os.path.exists(fn), f"file {fn} already exists"
        assert not os.path.exists(uc_fn), f"file {fn} already exists"

    embedding = AdvancePromptTuning(vec, name)
    embedding.step = 0
    embedding.save(fn)

    uc_embedding = AdvancePromptTuning(uc_vec, uc_name)
    uc_embedding.step = 0
    uc_embedding.save(fn)

    return fn, uc_fn


def validate_train_inputs(model_name, learn_rate, cfg_scale, data_root, template_file, steps, save_model_every, create_image_every, log_directory, name="embedding"):
    assert model_name, f"{name} not selected"
    assert learn_rate, "Learning rate is empty or 0"
    assert cfg_scale, "Negtive scale is empty or 0"
    assert data_root, "Dataset directory is empty"
    assert os.path.isdir(data_root), "Dataset directory doesn't exist"
    assert os.listdir(data_root), "Dataset directory is empty"
    assert template_file, "Prompt template file is empty"
    assert os.path.isfile(template_file), "Prompt template file doesn't exist"
    assert steps, "Max steps is empty or 0"
    assert isinstance(steps, int), "Max steps must be integer"
    assert steps > 0 , "Max steps must be positive"
    assert isinstance(save_model_every, int), "Save {name} must be integer"
    assert save_model_every >= 0 , "Save {name} must be positive or 0"
    assert isinstance(create_image_every, int), "Create image must be integer"
    assert create_image_every >= 0 , "Create image must be positive or 0"
    if save_model_every or create_image_every:
        assert log_directory, "Log directory is empty"


def train_embedding(embedding_name, learn_rate, cfg_scale, data_root, template_file, steps, save_embedding_every, create_image_every,
                    classifier_path, log_directory, training_width, training_height, save_image_with_stored_embedding, preview_from_txt2img,
                    preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_index, preview_cfg_scale, preview_seed, preview_width, preview_height):
    
    save_embedding_every = save_embedding_every or 0
    create_image_every = create_image_every or 0

    validate_train_inputs(embedding_name, learn_rate, cfg_scale, data_root, template_file, steps, save_embedding_every, create_image_every, 
                   log_directory, embedding_name)

    shared.state.textinfo = "Initializing textual inversion training..."
    shared.state.job_count = steps

    filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), embedding_name)

    if save_embedding_every > 0:
        embedding_dir = os.path.join(log_directory, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
    else:
        embedding_dir = None


    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    if create_image_every > 0 and save_image_with_stored_embedding:
        images_embeds_dir = os.path.join(log_directory, "image_embeddings")
        os.makedirs(images_embeds_dir, exist_ok=True)
    else:
        images_embeds_dir = None

    cond_model = shared.sd_model.cond_stage_model
    unload = shared.opts.unload_models_when_training

    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    with torch.autocast("cuda"):
        ds = apt_dataset.PersonalizedBase(data_root=data_root, placeholder_token=embedding_name,
                            model=shared.sd_model, device=devices.device, template_file=template_file, width=training_width, height=training_height)

    if unload:
        shared.sd_model.first_stage_model.to(devices.cpu)

    hijack = sd_hijack.model_hijack

    embedding = hijack.embedding_db.word_embeddings[embedding_name]
    embedding_uc = hijack.embedding_db.word_embeddings[embedding_name+'-uc']
    embedding.vec.requires_grad = True
    embedding_uc.vec.requires_grad = True

    optimizer = torch.optim.AdamW([embedding.vec, embedding_uc.vec], lr=learn_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.3)

    disc = XPDiscriminator(classifier_path) if (classifier_path is not None) and os.path.exists(classifier_path) else None

    if disc is not None:
        disc_label=torch.tensor([1]).cuda()
        ce=torch.nn.CrossEntropyLoss()
        print('use convnext discriminator')

    losses = torch.zeros((32,))

    last_saved_file = "<none>"
    last_saved_image = "<none>"

    ititial_step = embedding.step or 0
    if ititial_step > steps:
        return embedding, filename

    pbar = tqdm.tqdm(enumerate(ds), total=steps-ititial_step)
    for i, entries in pbar:
        timg, x, text = entries
        embedding.step = i + ititial_step
        embedding_uc.step = i + ititial_step

        if embedding.step > steps:
            break

        if shared.state.interrupted:
            break

        with torch.autocast("cuda"):
            c = cond_model([text])
            uc = cond_model([text.replace(ds.placeholder_token, ds.placeholder_token+'-uc')])

            c_in = torch.cat([uc, c])
            #print(c_in.shape)

            x = x.to(devices.device)
            output = shared.sd_model(x.unsqueeze(0), c_in, scale = cfg_scale)
            #print(shared.sd_model)
            x_samples_ddim = shared.sd_model.decode_first_stage(output[2])

            if disc is not None:
                #loss = ce(disc.get_all(x_samples_ddim), disc_label)
                loss = (1-disc.get_score(x_samples_ddim)).mean()
            else:
                loss = output[0] + torch_functional.l1_loss(timg, x_samples_ddim)

            del x

            losses[embedding.step % losses.shape[0]] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_num = embedding.step // len(ds)
        epoch_step = embedding.step % len(ds)

        pbar.set_description(f"[Epoch {epoch_num}: {epoch_step+1}/{len(ds)}]loss: {losses.mean():.7f}")

        steps_done = embedding.step + 1

        if embedding_dir is not None and steps_done % save_embedding_every == 0:
            last_saved_file = os.path.join(embedding_dir, f'{embedding_name}-{embedding.step}.pt')
            embedding.save(last_saved_file)
            last_saved_file = os.path.join(embedding_dir, f'{embedding_name}-uc-{embedding.step}.pt')
            embedding_uc.save(last_saved_file)

        write_loss(log_directory, "textual_inversion_loss.csv", embedding.step, len(ds), {
            "loss": f"{losses.mean():.7f}",
            "learn_rate": scheduler.get_last_lr()
        })

        if images_dir is not None and steps_done % create_image_every == 0:
            forced_filename = f'{embedding_name}-{steps_done}'
            last_saved_image = os.path.join(images_dir, forced_filename)

            shared.sd_model.first_stage_model.to(devices.device)

            p = processing.StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                do_not_save_grid=True,
                do_not_save_samples=True,
                do_not_reload_embeddings=True,
            )

            if preview_from_txt2img:
                p.prompt = preview_prompt
                p.negative_prompt = preview_negative_prompt
                p.steps = preview_steps
                p.sampler_index = preview_sampler_index
                p.cfg_scale = preview_cfg_scale
                p.seed = preview_seed
                p.width = preview_width
                p.height = preview_height
            else:
                p.prompt = entries[0].cond_text
                p.steps = 20
                p.width = training_width
                p.height = training_height

            preview_text = p.prompt

            processed = processing.process_images(p)
            image = processed.images[0]

            if unload:
                shared.sd_model.first_stage_model.to(devices.cpu)

            shared.state.current_image = image

            if save_image_with_stored_embedding and os.path.exists(last_saved_file) and embedding_yet_to_be_embedded:

                last_saved_image_chunks = os.path.join(images_embeds_dir, f'{embedding_name}-{steps_done}.png')

                info = PngImagePlugin.PngInfo()
                data = torch.load(last_saved_file)
                info.add_text("sd-ti-embedding", embedding_to_b64(data))

                title = "<{}>".format(data.get('name', '???'))

                try:
                    vectorSize = list(data['string_to_param'].values())[0].shape[0]
                except Exception as e:
                    vectorSize = '?'

                checkpoint = sd_models.select_checkpoint()
                footer_left = checkpoint.model_name
                footer_mid = '[{}]'.format(checkpoint.hash)
                footer_right = '{}v {}s'.format(vectorSize, steps_done)

                captioned_image = caption_image_overlay(image, title, footer_left, footer_mid, footer_right)
                captioned_image = insert_image_data_embed(captioned_image, data)

                captioned_image.save(last_saved_image_chunks, "PNG", pnginfo=info)
                embedding_yet_to_be_embedded = False

            last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt, shared.opts.samples_format, processed.infotexts[0], p=p, forced_filename=forced_filename, save_to_dirs=False)
            last_saved_image += f", prompt: {preview_text}"

        shared.state.job_no = embedding.step

        shared.state.textinfo = f"""
<p>
Loss: {losses.mean():.7f}<br/>
Step: {embedding.step}<br/>
Last prompt: {html.escape(text)}<br/>
Last saved embedding: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""

    checkpoint = sd_models.select_checkpoint()

    embedding.sd_checkpoint = checkpoint.hash
    embedding.sd_checkpoint_name = checkpoint.model_name
    embedding.cached_checksum = None
    embedding.save(filename)

    embedding_uc.sd_checkpoint = checkpoint.hash
    embedding_uc.sd_checkpoint_name = checkpoint.model_name
    embedding_uc.cached_checksum = None
    embedding_uc.save(f'{filename[:-3]}-uc.pt')

    return embedding, filename

