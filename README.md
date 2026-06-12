# Smol-VLM2 — Fine-tuning SmolVLM2 for Structured Food/Drink Extraction

Fine-tune the small vision-language model **[SmolVLM2-500M](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)** to look at an image and return **structured JSON** describing any food/drink in it.

> **Goal:** any image → `{is_food, image_title, food_items, drink_items}`

```json
{
  "is_food": 1,
  "image_title": "cheese plate",
  "food_items": ["toast", "herb garnish", "dipping sauce", "cheese wedge"],
  "drink_items": []
}
```

The base model fails to follow this structure; after supervised fine-tuning (SFT) the model sticks to the schema reliably.

## Repository contents

| Path | Description |
|---|---|
| `smolvlm2_finetune_tutorial.ipynb` | End-to-end tutorial: data formatting → SFT training → evaluation → HF Hub upload. |
| `demos/FoodExtract-Vision-v1/` | Gradio app (`app.py`) comparing the base vs. fine-tuned model side by side, plus `requirements.txt` and Space `README.md`. |

> **Note:** The trained model checkpoint (~1GB) is **not** stored in this repo — it lives on the HuggingFace Hub (see links below) and is excluded via `.gitignore`.

## Pipeline overview

1. **Load dataset** — [`mrdbourke/FoodExtract-1k-Vision`](https://huggingface.co/datasets/mrdbourke/FoodExtract-1k-Vision) (1k food images from Food101 + 500 non-food images). Each sample is an `image` + a target `output_label_json`.
2. **Format to chat** — `format_data()` converts each `(image, JSON)` pair into a `system` / `user` / `assistant` conversation:
   - `system` — sets the model's role (expert food/drink extractor).
   - `user` — the image + a prompt that enforces the output JSON schema.
   - `assistant` — the target `output_label_json` (the label to learn).
3. **Train** — `trl.SFTTrainer` with the `vision_model` backbone frozen (only the LLM is trained), following the [SmolDocling paper](https://arxiv.org/abs/2503.11576).
4. **Evaluate** — compare the base model vs. the fine-tuned model on held-out validation images.
5. **Ship** — upload the model and a Gradio demo to the HuggingFace Hub.

## Models & data

- **Base model:** https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- **Dataset:** https://huggingface.co/datasets/mrdbourke/FoodExtract-1k-Vision
- **Fine-tuned model (reference):** https://huggingface.co/mrdbourke/FoodExtract-Vision-SmolVLM2-500M-fine-tune-v1

## Running the notebook

The notebook is set up for a local GPU environment (developed on an NVIDIA DGX Spark / GB10). It installs its own extra dependencies (`trl`, `gradio`) on first run. A CUDA-capable GPU is recommended.

## Running the demo locally

```bash
cd demos/FoodExtract-Vision-v1
pip install -r requirements.txt
# edit FINE_TUNED_MODEL_ID in app.py to point at your own fine-tuned model
python app.py
```

## Possible next steps

- **Drop the input prompt** — train image → JSON directly to save inference tokens.
- **More real-world data** — current training set is small (1k food + 500 non-food).
- **Reduce repetitive generation** — the model can occasionally loop on a token (e.g. `"onions", "onions", ...`).

## Credits

Based on the tutorial by [mrdbourke / learn-huggingface](https://github.com/mrdbourke/learn-huggingface/blob/main/notebooks/hugging_face_vlm_fine_tune_tutorial.ipynb), adapted for a local GB10 environment.

## License

Apache-2.0
