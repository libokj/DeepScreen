from pathlib import Path

import pyrootutils
import gradio as gr

# Use this in a notebook
root = Path.cwd()

# Path finding with pyrootutils only works within a .py file
#
#
# root = pyrootutils.setup_root(
#     search_from=__file__,
#     indicator=[".git", "pyproject.toml"],
#     pythonpath=True,
#     dotenv=True,
# )

drug_encoder_list = [f.stem for f in root.parent.joinpath("configs/model/drug_encoder").iterdir() if f.suffix == ".yaml"]

drug_featurizer_list = [f.stem for f in root.parent.joinpath("configs/model/drug_featurizer").iterdir() if f.suffix == ".yaml"]

protein_encoder_list = [f.stem for f in root.parent.joinpath("configs/model/protein_encoder").iterdir() if f.suffix == ".yaml"]

protein_featurizer_list = [f.stem for f in root.parent.joinpath("configs/model/protein_featurizer").iterdir() if f.suffix == ".yaml"]

classifier_list = [f.stem for f in root.parent.joinpath("configs/model/classifier").iterdir() if f.suffix == ".yaml"]

preset_list = [f.stem for f in root.parent.joinpath("configs/model/preset").iterdir() if f.suffix == ".yaml"]


from typing import Optional

def drug_target_interaction(
        binary: bool,
        drug_encoder,
        drug_featurizer,
        protein_encoder,
        protein_featurizer,
        classifier,
        preset,) -> Optional[float]:


    return 1

def drug_encoder(
        binary: bool,
        drug_encoder,
        drug_featurizer,
        protein_encoder,
        protein_featurizer,
        classifier,
        preset,):

    return

def protein_encoder(
        binary: bool,
        drug_encoder,
        drug_featurizer,
        protein_encoder,
        protein_featurizer,
        classifier,
        preset,):

    return

# demo = gr.Interface(
#     fn=drug_target_interaction,
#     inputs=[
#         gr.Radio(["True", "False"]),
#         gr.Dropdown(drug_encoder_list),
#         gr.Dropdown(drug_featurizer_list),
#         gr.Dropdown(protein_encoder_list),
#         gr.Dropdown(protein_featurizer_list),
#         gr.Dropdown(classifier_list),
#         gr.Dropdown(preset_list),
#     ],
#     outputs=["number"],
#     show_error=True,
#
# )
#
# demo.launch()


from omegaconf import DictConfig, OmegaConf

type_to_component_map = {list: gr.Text, int: gr.Number, float: gr.Number}


def get_config_choices(config_path: str):
    return [f.stem for f in Path("../configs/", config_path).iterdir() if f.suffix == ".yaml"]


def create_blocks_from_config(cfg: DictConfig):
    with gr.Blocks() as blocks:
        for key, value in cfg.items():
            if type(value) in [int, float]:
                component = gr.Number(value=value, label=key, interactive=True)
            if type(value) in [dict, DictConfig]:
                with gr.Tab(label=key):
                    component = create_blocks_from_config(value)
            else:
                component = gr.Text(value=value, label=key, interactive=True)
    return blocks


def create_interface_from_config(fn: callable, cfg: DictConfig):
    inputs = []

    for key, value in OmegaConf.to_object(cfg).items():
        component = type_to_component_map.get(type(value), gr.Text)
        inputs.append(component(value=value, label=key, interactive=True))

    interface = gr.Interface(fn=fn, inputs=inputs, outputs="label")

    return interface


import hydra

with hydra.initialize(version_base=None, config_path="../configs/"):
    cfg = hydra.compose("train")