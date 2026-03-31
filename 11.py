import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)
prompt = input("Enter text prompt: ")
result = pipe(prompt)
image = result.images[0]
image.save("generated_image.png")
plt.imshow(image)
plt.axis("off")
plt.title("Generated Image")
plt.show()
print("Image saved as generated_image.png")
