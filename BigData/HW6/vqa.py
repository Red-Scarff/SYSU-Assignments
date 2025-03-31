from PIL import Image
import requests
from transformers import ViltProcessor, ViltForQuestionAnswering

# 1. 加载预训练模型和处理器
model_name = "dandelin/vilt-b32-finetuned-vqa"  # ViLT的VQA专用模型
processor = ViltProcessor.from_pretrained(model_name)
model = ViltForQuestionAnswering.from_pretrained(model_name)
model.to("cuda")
# 2. 准备输入数据
# 方式1：从网络加载图片
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# 方式2：本地图片
image = Image.open("test.jpg")

# 定义问题
question = "How many players are there in this picture?"

# 3. 预处理（图像+文本编码）
encoding = processor(image, question, return_tensors="pt")
encoding = {k: v.to("cuda") for k, v in encoding.items()}
# 4. 模型推理
outputs = model(**encoding)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()

# 5. 解析答案
answer = model.config.id2label[predicted_class_idx]
print(f"问题: {question}")
print(f"答案: {answer}")