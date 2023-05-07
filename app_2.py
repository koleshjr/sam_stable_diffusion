import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from consts import device, sam_checkpoint, model_type, stable_diffusion_inpaint_model


# initialize sam
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# initialize stable diffusion inpainting pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    stable_diffusion_inpaint_model,
    torch_dtype=torch.float16,  # save some memory
)
pipe = pipe.to(device)
selected_pixels = []


with gr.Blocks() as demo:
    with gr.Row():
        original_img = gr.Image(label="Original Image")
        wig_img = gr.Image(label="Wig to Buy")
        mask_img = gr.Image(label="Wig Mask")
        output_img = gr.Image(label="Generated Image")

    with gr.Row():
        prompt_text = gr.Textbox(lines=1, label="Prompt")

    with gr.Row():
        submit = gr.Button("Submit")

    # generate mask of the wig
    def generate_mask(image, evt: gr.SelectData):
        selected_pixels.append(evt.index)
        predictor.set_image(image)
        input_points = np.array(selected_pixels)
        input_labels = np.ones(input_points.shape[0])
        mask, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
        mask = np.logical_not(mask)  # mask is a binary array, select everything on the foreground and background will change
        mask = Image.fromarray(mask[0, :, :])
        return mask

    # inpainting function
    def inpaint(original_image, wig_mask, prompt):
        original_image = Image.fromarray(original_image)
        wig_mask = Image.fromarray(wig_mask)

        original_image = original_image.resize((512, 512))
        wig_mask = wig_mask.resize((512, 512))

        output = pipe(prompt=prompt,
                      image=original_image,
                      mask_image=wig_mask).images[0]

        return output

    original_img.select(generate_mask, [wig_img], [mask_img])
    submit.click(inpaint,
                 inputs=[original_img, mask_img, prompt_text],
                 outputs=[output_img])

    if __name__ == "__main__":
        demo.launch()
