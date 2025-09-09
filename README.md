# pkumi
北京大学艺术学院开发的多模态大模型的艺术品审美评价排行榜

# 🖼️ AI 艺术评论家：多模型对比审美评价系统

> 用 AI 解读中国书画之美 —— 基于多模态大模型的自适应审美评测平台

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)
[![Flask](https://img.shields.io/badge/framework-Flask-ff69b4)](https://flask.palletsprojects.com/)
[![OpenAI Compatible](https://img.shields.io/badge/API-Compatible-9cf)](https://openai.com)

本项目构建了一个**多模型、可对比、自适应抽样**的艺术品审美评价系统，支持对博物馆级中国书画作品进行深度美学分析。系统集成了来自字节、腾讯、OpenAI、Anthropic、Google、阿里通义、清华智谱等厂商的 **20+ 视觉语言模型（VLM）**，通过匿名与非匿名双模式评测，探索 AI 在艺术理解与审美判断上的能力边界。

---

## 🌟 核心特性

✅ **多模型集成**  
支持 `Doubao`、`Claude`、`Gemini`、`GPT-4`、`GLM-4V`、`Qwen-VL`、`Hunyuan` 等国内外主流模型，涵盖 ByteDance、OpenRouter、Tencent 三大 API 平台。

✅ **双模式评价机制**  
- **实名模式**：提供作品名称、作者等元数据，测试模型的综合艺术理解能力。  
- **匿名模式**：仅基于图像进行“盲评”，测试模型的纯粹视觉审美能力。

✅ **自适应模型抽样器**  
基于 [Chatbot Arena](https://arena.lmsys.org) 论文思想，实现动态抽样策略：
- 优先选择**胜负接近**的模型对（高不确定性）
- 优先探索**对战次数少**的组合（高探索性）
- 自动生成技术性选择理由，提升可解释性

✅ **结构化美学分析框架**  
所有模型输出遵循统一的 Markdown 评论模板，包含六大维度：
1. 审美价值概述（结合年代与流派）
2. 内容客观描述
3. 造型、构图、色彩分析
4. 材质、工艺、结构之美
5. 意象、氛围与意境
6. 综合审美价值判断

✅ **用户反馈闭环**
- 支持用户对两个模型的评论进行投票
- 收集文本反馈与错误报告
- 数据持久化至 `ratings.csv`、`feedback.csv`、`error_reports.csv`

✅ **前端交互体验**
- 响应式画廊页面，支持按年代筛选（唐前、宋元、明、清、近现代）
- 作品详情页与 AI 评论对比界面
- 支持本地部署与调试（Flask + Jinja2）

---

## 📦 技术架构
