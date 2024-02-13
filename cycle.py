from diffusers import DiffusionPipeline
import torch
from transformers import pipeline


pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

prompt = "An astronaut riding a green horse"

images = pipe(prompt=prompt).images[0]
images.save("output.png")

pipe = pipeline("image-to-text", model="Salesforce/blip2-opt-2.7b")
caption = pipe(images="output.png")[0]['generated_text']
print(caption)
