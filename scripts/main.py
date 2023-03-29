import ast
from collections.abc import Iterable
import os
import sys

import gradio as gr
import numpy as np
from numpy.typing import NDArray

from modules import images, paths, script_callbacks, scripts, shared

try:
    from scripts.effects import effects, pixel_sort
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
    from effects import effects, pixel_sort

class EffectOptions:
    def __init__(self, name: str, func: callable):
        self.name = name
        self.func = func

    def process(self, img: NDArray, args):
        return self.func(img, *args)

effects = [
    EffectOptions("Nothing", lambda img, _: img),
    EffectOptions("Gaussian Blur", effects.gaussian_blur),
    EffectOptions("Dither", effects.dither),
    EffectOptions("Sharpen", effects.sharpen),
    EffectOptions("Pixel Sort", pixel_sort.pixel_sort),
]

def on_ui_settings():
    section = ('post-processing', "Post Processing")

    # [setting_name], [default], [label], [component], [component_args]
    settings = [
        ('post_processing_num_effects', 3, "Number of Effects (Reload require)"),
    ]

    for setting_name, *setting_options in settings:
        shared.opts.add_option(setting_name, shared.OptionInfo(*setting_options, section=section))

    pass

def stacked_effects_tab():
    with gr.Tab("Main"):
        with gr.Row():
            base_img = gr.Image().style(height=512)
            output_img = gr.Image(None).style(height=512)

        num_effects = int(shared.opts.post_processing_num_effects)

        effect_dropdowns = []
        effect_params = []
        effect_args = []

        def func_doc(x):
            return effects[x].func.__doc__

        for i in range(num_effects):
            with gr.Row():
                with gr.Column(variant="panel"):
                    cur_effects_dropdown = gr.Dropdown(label=f"Effects {i + 1}", choices=[e.name for e in effects], value=effects[0].name, type="index")
                effect_dropdowns.append(cur_effects_dropdown)
                with gr.Column(variant="panel"):
                    cur_effects_params = gr.Markdown("No parameters")
                    effect_params.append(cur_effects_params)
                    cur_effects_args = gr.Textbox(label="Args", value="")
                    effect_args.append(cur_effects_args)

            cur_effects_dropdown.change(fn=func_doc, inputs=[cur_effects_dropdown], outputs=[cur_effects_params])

        process = gr.Button("Process")

        def parse_args(x):
            vals = ast.literal_eval(x) if x else ()
            return vals if isinstance(vals, Iterable) else (vals,)

        def process_fn(*inputs):
            img = inputs[-1]
            effect_indices = inputs[:num_effects]
            effect_args = inputs[num_effects:-1]

            for effect_index, args in zip(effect_indices, effect_args):
                if effect_index != 0:
                    img = effects[effect_index].process(img, parse_args(args))

            return img

        process.click(fn=process_fn, inputs=effect_dropdowns + effect_args + [base_img], outputs=[output_img])

def workshop_tab():
    with gr.Tab("Workshop"):
        with gr.Row():
            copied_img = gr.Image().style(height=512)
            output_img = gr.Image(None).style(height=512)

        with gr.Tabs(elem_id='effect_tabs'):
            with gr.Tab("Gaussian Blur"):
                with gr.Row():
                    kernel_size = gr.Slider(label="Kernel Size", minimum=1, maximum=21, step=2, value=5)
                    sigma = gr.Slider(label="Sigma", minimum=0, maximum=10, step=0.5, value=1.4)


        

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as post_processing:
        gr.Textbox("Post Processing", show_label=False)

        stacked_effects_tab()
        workshop_tab()
        
    return (post_processing, "Post Processing", "post_processing"),

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
