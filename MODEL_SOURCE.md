# 模型来源

- 初始权重：`yolo_carla_best.pt`
- 发布者：Jkdxbns
- Hugging Face：https://huggingface.co/jkdxbns/autonomous-driving-carla
- 关联GitHub：https://github.com/Jkdxbns/autonomous-driving-carla

初始权重属于第三方预训练模型，不是第18组训练的模型。本项目使用其目标检测器建立基线，然后基于自制晴天或雨天数据分别微调50轮。第三方初始权重和两份微调权重均未提交到本仓库。

关联开源项目声明采用MIT License。Ultralytics、CARLA及其他依赖仍适用各自许可条款。
