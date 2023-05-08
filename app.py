import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from consts import device, sam_checkpoint, model_type, stable_diffusion_inpaint_model


#initialize sam
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

#initialize stable diffusion impainting pipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
stable_diffusion_inpaint_model,
torch_dtype= torch.float16, #save some memory #use this when the device is gpu
)
pipe = pipe.to(device)
selected_pixels =[]

with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Input")
        mask_img = gr.Image(label="Mask")
        output_img = gr.Image(label="Output")

    with gr.Row():
        prompt_text = gr.Textbox(lines=1, label ="Prompt")

    with gr.Row():
        submit = gr.Button("Submit")

    #click input image, mask created, and that mask used for inpainting
    def generate_mask(image, evt: gr.SelectData):
        selected_pixels.append(evt.index)
        predictor.set_image(image)
        input_points = np.array(selected_pixels)
        input_labels = np.ones(input_points.shape[0])
        mask,_,_ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
        #(n, sz, sz)
        mask = np.logical_not(mask) #mask is a binary array, select everything on the foreground and backgrund will change
        mask = Image.fromarray(mask[0, :, :])
        return mask



    #inpainting function,
    def inpaint(image, mask, prompt):
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        image = image.resize((512,512))
        mask = mask.resize((512,512))

        output = pipe(prompt=prompt, 
                      image=image, 
                      mask_image = mask).images[0]
        
        return output
    
    input_img.select(generate_mask, [input_img], [mask_img])
    submit.click(inpaint, 
                 inputs=[input_img, mask_img, prompt_text],
                 outputs=[output_img])
    
if __name__ == "__main__":
    demo.launch()

    



