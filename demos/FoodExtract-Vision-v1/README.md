---
title: FoodExtract-Vision Fine-tuned VLM Structued Data Extractor
emoji: 🍟➡️📝
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

Fine-tuned SmolVLM2-500M to extract food and drink items from images.

Input can be any kind of image and output will be a formatted string such as the following:

```json
{'is_food': 0, 'image_title': '', 'food_items': [], 'drink_items': []}
```

Or for an image of food:

```json
{'is_food': 1, 'image_title': 'fried calamari', 'food_items': ['fried calamari'], 'drink_items': []}
```

Note: This README.md was authored in a live tutorial recorded for YouTube (https://youtu.be/_EMfJSmLSKE?si=qI2U4sRg45XQQlSW).

Huggingface Space App : https://huggingface.co/spaces/jhkim3217/FoodExtract-Vision-v1/commit/458fe90161cdeb2c30e3408bc528be97953538f2


