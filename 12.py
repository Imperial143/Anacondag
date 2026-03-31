from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora()
prompt = "A diseased tomato leaf with brown spots, agricultural crop disease, high detail"
image = pipe(prompt, num_inference_steps=10, guidance_scale=1.0).images[0]
plt.imshow(image)
plt.axis("off")
plt.title("Generated Agricultural Disease Image")
plt.show()
