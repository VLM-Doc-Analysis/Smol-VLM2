import torch
import gradio as gr

import spaces
from transformers import pipeline

BASE_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
# ⚠️ 아래 모델 ID를 본인의 fine-tuned 모델 경로로 변경하세요
FINE_TUNED_MODEL_ID = "mrdbourke/FoodExtract-Vision-SmolVLM2-500M-fine-tune-v1"
OUTPUT_TOKENS = 256

# Load original base model (no fine-tuning)
print(f"[INFO] Loading Original Model")
original_pipeline = pipeline(
    "image-text-to-text",
    model=BASE_MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto"
)

# Load fine-tuned model
print(f"[INFO] Loading Fine-tuned Model")
ft_pipe = pipeline(
    "image-text-to-text",
    model=FINE_TUNED_MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto"
)

def create_message(input_image):
    return [{'role': 'user',
 'content': [{'type': 'image',
   'image': input_image},
  {'type': 'text',
   'text': "Classify the given input image into food or not and if edible food or drink items are present, extract those to a list. If no food/drink items are visible, return empty lists.\n\nOnly return valid JSON in the following form:\n\n```json\n{\n  'is_food': 0, # int - 0 or 1 based on whether food/drinks are present (0 = no foods visible, 1 = foods visible)\n  'image_title': '', # str - short food-related title for what foods/drinks are visible in the image, leave blank if no foods present\n  'food_items': [], # list[str] - list of visible edible food item nouns\n  'drink_items': [] # list[str] - list of visible edible drink item nouns\n}\n```\n"}]}]

@spaces.GPU
def extract_foods_from_image(input_image):
    input_image = input_image.resize(size=(512, 512))
    input_message = create_message(input_image=input_image)

    # Get outputs from base model (not fine-tuned)
    original_pipeline_output = original_pipeline(text=[input_message],
                                                 max_new_tokens=OUTPUT_TOKENS)

    outputs_pretrained = original_pipeline_output[0][0]["generated_text"][-1]["content"]

    # Get outputs from fine-tuned model (fine-tuned on food images)
    ft_pipe_output = ft_pipe(text=[input_message],
                             max_new_tokens=OUTPUT_TOKENS)
    outputs_fine_tuned = ft_pipe_output[0][0]["generated_text"][-1]["content"]

    return outputs_pretrained, outputs_fine_tuned

demo_title = "🥑➡️📝 FoodExtract-Vision with a fine-tuned SmolVLM2-500M"
demo_description = """* **Base model:** https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct
* **Fine-tuning dataset:** https://huggingface.co/datasets/mrdbourke/FoodExtract-1k-Vision (1k food images and 500 not food images)
* **Fine-tuned model:** https://huggingface.co/mrdbourke/FoodExtract-Vision-SmolVLM2-500M-fine-tune-v1

## Overview

Extract food and drink items in a structured way from images.

The original model outputs fail to capture the desired structure. But the fine-tuned model sticks to the output structure quite well.

However, the fine-tuned model could definitely be improved with respects to its ability to extract the right food/drink items.

Both models use the input prompt:

````
Classify the given input image into food or not and if edible food or drink items are present, extract those to a list. If no food/drink items are visible, return empty lists.

Only return valid JSON in the following form:

```json
{
  'is_food': 0, # int - 0 or 1 based on whether food/drinks are present (0 = no foods visible, 1 = foods visible)
  'image_title': '', # str - short food-related title for what foods/drinks are visible in the image, leave blank if no foods present
  'food_items': [], # list[str] - list of visible edible food item nouns
  'drink_items': [] # list[str] - list of visible edible drink item nouns
}
```
````

Except one model has been fine-tuned on the structured data whereas the other hasn't.

Notable next steps would be:
* **Remove the input prompt:** Just train the model to go straight from image -> text (no text prompt on input), this would save on inference tokens.
* **Fine-tune on more real-world data:** Right now the model is only trained on 1k food images (from Food101) and 500 not food (random internet images), training on real world data would likely significantly improve performance.
* **Fix the repetitive generation:** The model can sometimes get stuck in a repetitive generation pattern, e.g. "onions", "onions", "onions", etc. We could look into patterns to help reduce this.
"""

demo = gr.Interface(
    fn=extract_foods_from_image,
    inputs=gr.Image(type="pil"),
    title=demo_title,
    description=demo_description,
    outputs=[gr.Textbox(lines=4, label="Original Model (not fine-tuned)"),
             gr.Textbox(lines=4, label="Fine-tuned Model")],
    examples=[["examples/camera.jpeg"],
              ["examples/Tandoori-Chicken.jpg"],
              ["examples/fries.jpeg"]],
    # 시작 시 예제 캐싱 비활성화: 최신 Gradio가 캐싱 단계에서 로컬 예제 파일을
    # 차단(InvalidPathError)하는 문제 방지 + 무료 CPU Space 시작 부담 감소
    cache_examples=False,
)

if __name__ == "__main__":
    # allowed_paths: 로컬 예제 이미지(examples/)를 Gradio 캐시/서빙에 허용
    # (최신 Gradio의 InvalidPathError "... not uploaded by a user" 방지)
    demo.launch(share=False, allowed_paths=["examples"])
