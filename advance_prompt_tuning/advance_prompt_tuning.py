import os
import sys
import traceback

import torch
import tqdm
import html
import datetime
from torch.nn import functional as torch_functional

from modules import shared, devices, sd_hijack, processing, sd_models
import advance_prompt_tuning.dataset as apt_dataset
from modules.conv_next.interface import XPDiscriminator

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


def create_embedding(name, num_vectors_per_token, init_text='*'):
    cond_model = shared.sd_model.cond_stage_model
    embedding_layer = cond_model.wrapped.transformer.text_model.embeddings

    ids = cond_model.tokenizer(init_text, max_length=num_vectors_per_token, return_tensors="pt", add_special_tokens=False)["input_ids"]
    embedded = embedding_layer.token_embedding.wrapped(ids.to(devices.device)).squeeze(0)
    vec = torch.zeros((num_vectors_per_token, embedded.shape[1]), device=devices.device)

    for i in range(num_vectors_per_token):
        vec[i] = embedded[i * int(embedded.shape[0]) // num_vectors_per_token]

    fn = os.path.join(shared.cmd_opts.embeddings_dir, f"{name}.pt")
    assert not os.path.exists(fn), f"file {fn} already exists"

    embedding = AdvancePromptTuning(vec, name)
    embedding.step = 0
    embedding.save(fn)

    return fn


def train_embedding(embedding_name, learn_rate, cfg_scale, data_root, log_directory, steps, create_image_every, save_embedding_every, template_file,
                    classifier_path, img_size):
    assert embedding_name, 'embedding not selected'

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

    cond_model = shared.sd_model.cond_stage_model

    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    with torch.autocast("cuda"):
        ds = apt_dataset.PersonalizedBase(data_root=data_root, size=img_size, placeholder_token=embedding_name,
                            model=shared.sd_model, device=devices.device, template_file=template_file, width=img_size, height=img_size)

    hijack = sd_hijack.model_hijack

    embedding = hijack.embedding_db.word_embeddings[embedding_name]
    embedding_uc = hijack.embedding_db.word_embeddings[embedding_name+'-uc']
    embedding.vec.requires_grad = True
    embedding_uc.vec.requires_grad = True

    optimizer = torch.optim.AdamW([embedding.vec, embedding_uc.vec], lr=learn_rate)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.3)

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
    for i, (timg, x, text) in pbar:
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
            schedule.step()

        pbar.set_description(f"loss: {losses.mean():.7f}, grad:{embedding.vec.grad.mean():.7f}")

        if embedding.step > 0 and embedding_dir is not None and embedding.step % save_embedding_every == 0:
            last_saved_file = os.path.join(embedding_dir, f'{embedding_name}-{embedding.step}.pt')
            embedding.save(last_saved_file)
            last_saved_file = os.path.join(embedding_dir, f'{embedding_name}-uc-{embedding.step}.pt')
            embedding_uc.save(last_saved_file)

        if embedding.step > 0 and images_dir is not None and embedding.step % create_image_every == 0:
            last_saved_image = os.path.join(images_dir, f'{embedding_name}-{embedding.step}.png')

            p = processing.StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                prompt=text,
                steps=20,
                do_not_save_grid=True,
                do_not_save_samples=True,
                negative_prompt=text.replace(ds.placeholder_token, ds.placeholder_token+'-uc'),
                cfg_scale=cfg_scale,
            )

            processed = processing.process_images(p)
            image = processed.images[0]

            shared.state.current_image = image
            image.save(last_saved_image)

            last_saved_image += f", prompt: {text}"

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

