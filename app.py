import time
from collections import defaultdict

import gradio as gr
from pydantic import BaseModel

import classifier as clf


async def make_preds(*args):
    num_args = len(args)
    entity_names = args[: int(num_args // 2)]
    dirs = args[int(num_args // 2) : -1]
    test_dir = args[-1]
    training_contents = []
    prompt = "I am going to show you pictures of some entities, for each picture, I will provide the entity's name."
    for e, d in zip(entity_names, dirs):
        training_contents.extend(
            await clf.create_training_contents(entity_name=e, entity_directory_name=d)
        )
    training_contents = [prompt] + training_contents
    model: BaseModel = clf.create_dynamic_model(entity_values=entity_names)
    preds = await clf.test(test_dir, training_contents, model)
    final_preds = defaultdict(list)
    for p, f in zip(preds, test_dir):
        for e in p.entity:
            final_preds[e].append(f)

    return final_preds, gr.Markdown(height=100)


with gr.Blocks() as demo:

    preds = gr.State({})
    entity_names = []
    dirs = []

    number_of_entities = gr.Radio(
        choices=[1, 2, 3, 4, 5], label="Enter the number of your entities", value=1
    )

    @gr.render(inputs=number_of_entities)
    def generating_ui(number_of_entities):
        global entity_names
        global dirs

        entity_names = []
        dirs = []
        with gr.Row():
            for num in range(number_of_entities):
                with gr.Column():
                    entity_names.append(
                        gr.Textbox(label=f"Enter entity name number {num+1}")
                    )
                    dirs.append(
                        gr.File(
                            label=f"Upload entity Image number {num+1}",
                            file_count="directory",
                        )
                    )
        gr.Markdown("Upload a folder containing your test images:")
        test_dir = gr.File(label="Test directory", file_count="directory")
        markdown = gr.Markdown(height=100)
        submit_btn.click(
            fn=make_preds,
            inputs=[*entity_names, *dirs, test_dir],
            outputs=[preds, markdown],
            show_progress="full",
        )

    submit_btn = gr.Button(value="Submit")

    @gr.render(inputs=[preds])
    def render_preds(preds, _):
        if preds:
            with gr.Row():
                for entity, images in preds.items():
                    gr.Gallery(label=entity, value=images)


demo.launch()
