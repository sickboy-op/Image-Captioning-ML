
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


kirayeKAprocessor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


img_path = "render.jpg"  
raw_image = Image.open(img_path).convert("RGB")

image = kirayeKAprocessor(raw_image, return_tensors="pt")
out = model.generate(**image)
caption = kirayeKAprocessor.decode(out[0], skip_special_tokens=True)

print("\n Generated Caption from the picture is :", caption)
