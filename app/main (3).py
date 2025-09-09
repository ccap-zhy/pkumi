# ==============================================================================
# 文件: app.py (V6.1 - 修复NameError版)
# ==============================================================================

# --- 导入所需库 ---
import os
import base64
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory, abort
from flask_cors import CORS
from openai import OpenAI
from typing import Dict # <--- 就是增加了这一行！
import logging
import random
import uuid
from datetime import datetime
from sampler import ModelSampler

# --- V6 更新: 使用绝对路径，让服务更健壮 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Flask 应用和配置 ---
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
app.config['JSON_AS_ASCII'] = False
CORS(app)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- 全局配置 ---
API_KEY = "sk-2NDVAyUmnp0j6xtZYTr9COIZotrFHoEd0D74LnVOyAsGqv5D"
BASE_URL = "https://www.dmxapi.cn/v1"
# 新增的OpenRouter API配置
OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- 创建两个不同的客户端 ---
# 原有的客户端
original_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
# OpenRouter客户端
openrouter_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
tengxun_client = OpenAI(
    api_key="sk-NMJW1J9COjwZwqOUJMWmHPhcelm55W0XrmzNsiPKE5LgQOEa",  # 混元 APIKey
    base_url="https://api.hunyuan.cloud.tencent.com/v1", # 混元 endpoint
)
DATA_FILE_PATH = os.path.join(BASE_DIR, "中国博物馆书画数据目录.xlsx")
IMAGE_DIRECTORY = os.path.join(BASE_DIR, "images")
RATINGS_FILE_PATH = os.path.join(BASE_DIR, "ratings.csv")
FEEDBACK_FILE_PATH = os.path.join(BASE_DIR, "feedback.csv") # <-- 新增：独立的反馈文件路径

ERROR_REPORT_FILE_PATH = os.path.join(BASE_DIR, "error_reports.csv") # 报错反馈

# --- 数据加载 ---
def map_era_to_group(era):
    """将详细年代映射到指定的筛选分组"""
    era_str = str(era) # 确保输入是字符串
    # 唐前
    if any(keyword in era_str for keyword in ['唐',  '晋', '南北朝', '隋', '敦煌', '汉', '辽', '战国', '周', '春秋', '三国']):
        return '唐前'
    # 宋元
    if any(keyword in era_str for keyword in ['宋', '元', '金']):
        return '宋元'
    # 明
    if any(keyword in era_str for keyword in ['明']):
        return '明'
    # 清
    if '清' in era_str:
        return '清'
    # 近现代
    if any(keyword in era_str for keyword in ['近现代', '当代']):
        return '近现代'
    # 其他不在此分类中
    return '其他'

try:
    datas = pd.read_excel(DATA_FILE_PATH)
    datas["path"] = "/images/" + datas["id"] + ".jpg"
    # --- 新增：应用映射函数，创建新的'era_group'列 ---
    datas['era_group'] = datas['年代'].apply(map_era_to_group)
    print(f"✅ [服务端] 成功加载 {len(datas)} 条艺术品数据，并完成年代分组。")
except FileNotFoundError:
    print(f"❌ [服务端] 错误: 数据文件 '{DATA_FILE_PATH}' 未找到。")
    datas = None
datas.loc[datas["收藏地"].isna(),"收藏地"] = "未记录"

datas.loc[datas['材质'].isna(),'材质'] = "未记录"
datas.loc[datas['形制'].isna(),'形制'] = "未记录"
datas.loc[datas['材料'].isna(),'材料'] = "未记录"
datas = datas[~datas["年代"].str.contains("日本|室町|五代|不详")]




# --- 完整的模型配置 ---
MODEL_CONFIG = {
    # 原有的模型（使用原API）
    "Doubao-1.5-vision-pro-32k": {
        "client": original_client, 
        "model_name": "Doubao-1.5-vision-pro-32k", 
        "provider": "ByteDance"
    },
    "doubao-seed-1-6-thinking-250615": {
        "client": original_client, 
        "model_name": "doubao-seed-1-6-thinking-250615", 
        "provider": "ByteDance"
    },
    "openai/o3": {
        "client": original_client, 
        "model_name": "o3", 
        "provider": "Openai"
    },
    "claude-sonnet-4-20250514-thinking": {
        "client": original_client, 
        "model_name": "claude-sonnet-4-20250514-thinking", 
        "provider": "Claude"
    },
    "qvq-max-2025-03-25": {
        "client": original_client, 
        "model_name": "qvq-max-2025-03-25", 
        "provider": "Qwen"
    },
    "step-1o-vision-32k": {
        "client": original_client, 
        "model_name": "step-1o-vision-32k", 
        "provider": "Step"
    },
    # 新增的OpenRouter模型
    "thudm/glm-4.1v-9b-thinking": {
        "client": openrouter_client, 
        "model_name": "thudm/glm-4.1v-9b-thinking", 
        "provider": "OpenRouter"
    },
    "anthropic/claude-sonnet-4": {
        "client": openrouter_client, 
        "model_name": "anthropic/claude-sonnet-4", 
        "provider": "OpenRouter"
    },
    "anthropic/claude-opus-4": {
        "client": openrouter_client, 
        "model_name": "anthropic/claude-opus-4", 
        "provider": "OpenRouter"
    },
    "anthropic/claude-opus-4-20250514-thinking": {
        "client": original_client, 
        "model_name": "claude-opus-4-20250514-thinking", 
        "provider": "OpenRouter"
    },
    "google/gemini-2.5-flash": {
        "client": openrouter_client, 
        "model_name": "google/gemini-2.5-flash", 
        "provider": "OpenRouter"
    },
    "google/gemini-2.5-pro": {
        "client": openrouter_client, 
        "model_name": "google/gemini-2.5-pro", 
        "provider": "OpenRouter"
    },
    "anthropic/claude-3.7-sonnet:thinking": {
        "client": openrouter_client, 
        "model_name": "anthropic/claude-3.7-sonnet:thinking", 
        "provider": "OpenRouter"
    },
    "openai/gpt-4.1": {
        "client": openrouter_client, 
        "model_name": "openai/gpt-4.1", 
        "provider": "OpenRouter"
    },
    "meta-llama/llama-4-maverick": {
        "client": openrouter_client, 
        "model_name": "meta-llama/llama-4-maverick", 
        "provider": "OpenRouter"
    },
    "openai/o4-mini-high": {
        "client": openrouter_client, 
        "model_name": "openai/o4-mini-high", 
        "provider": "OpenRouter"
    },
    "x-ai/grok-4": {
        "client": openrouter_client, 
        "model_name": "x-ai/grok-4", 
        "provider": "OpenRouter"
    },
    "qwen/qwen2.5-vl-72b-instruct": {
        "client": openrouter_client, 
        "model_name": "qwen/qwen2.5-vl-72b-instruct", 
        "provider": "OpenRouter"
    },
    "tencent/hunyuan-t1-vision-20250619": {
        "client": tengxun_client, 
        "model_name": "hunyuan-t1-vision-20250619", 
        "provider": "OpenRouter"
    },
    
}
# --- 新增：初始化自适应模型抽样器 ---
try:
    model_sampler = ModelSampler(list(MODEL_CONFIG.keys()), RATINGS_FILE_PATH)
    print("✅ [服务端] 自适应模型抽样器已成功初始化。")
except Exception as e:
    print(f"❌ [服务端] 初始化模型抽样器失败: {e}")
    model_sampler = None



print(f"✅ [服务端] 已配置模型: {list(MODEL_CONFIG.keys())}")


# --- 核心分析逻辑 ---
def run_art_cot_analysis(model_key: str, artwork_info: pd.Series) -> Dict:
    model_details = MODEL_CONFIG.get(model_key)
    local_image_path = os.path.join(IMAGE_DIRECTORY, artwork_info['id'] + ".jpg")
    
    if not os.path.exists(local_image_path):
        return {"error": f"图片文件未找到: {local_image_path}"}
        
    base64_image = encode_image_to_base64(local_image_path)
    client = model_details["client"]
    model_name = model_details["model_name"]
    
    combined_prompt = (
        # ---------- 任务说明 ----------
    # f"你的核心任务是对这件艺术品（已知信息：名称《{artwork_info['名称']}》，作者/出处: {artwork_info['作者']}）进行一次深刻的审美评价，撰写一篇约1000字的艺术评论。"
    "你是一位资深的艺术史学家、艺术批评家和大学学者，拥有对经典艺术（包括绘画、雕塑、建筑、书法等）的卓越洞察力与品味。"
        "你的核心任务是对这件艺术品"
        "进行一次深刻而具洞见的分析和审美评价，全文约1500字。\n\n"
        "写作须遵循以下规范：\n"
        "• 使用 Markdown 格式。\n"
        "• 开头先为本文拟定一个专业的一级标题（以 “# ” 开头，可自由命名，但需契合作品气质）。\n"
        "• 之后严格依次包含六大板块：\n"
        "  1. **作品审美价值概述、要求概述时结合这个作品的年代和在中国艺术中的流派**（≈200 字）\n"
        "  2. **对作品的内容进行客观描述**\n"
        "  3. **分析造型、构图、色彩的美感**\n"
        "  4. **分析材质、工艺、结构上的美感**\n"
        "  5. **分析作品的意象、氛围与意境**\n"
        "  6. **对作品的审美价值进行判断**（≈100 字）\n"
        "• 评论重心应放在作品的艺术魅力与审美价值，而非功能、知名度、考古学分析或拍卖价格；所有论点都须与作品的具体元素紧密对应"
    )
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": [{"type": "text", "text": combined_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
        )
        return {
            "model_name": model_key,
            "response": response.choices[0].message.content,
            "model_info": { "name": model_name, "provider": model_details.get("provider", "Unknown") }
        }
    except Exception as e:
        return {"error": f"模型 {model_name} 在处理时发生错误: {str(e)}"}

# --- 辅助函数 ---
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def run_art_cot_analysis_anonymous(model_key: str, artwork_info: pd.Series) -> Dict:
    """匿名评价函数，不向模型提供作者和标题信息"""
    model_details = MODEL_CONFIG.get(model_key)
    local_image_path = os.path.join(IMAGE_DIRECTORY, artwork_info['id'] + ".jpg")
    
    if not os.path.exists(local_image_path):
        return {"error": f"图片文件未找到: {local_image_path}"}
        
    base64_image = encode_image_to_base64(local_image_path)
    client = model_details["client"]
    model_name = model_details["model_name"]
    
    # [匿名版] 提示词 - 核心区别在于第一句话，不提供任何已知信息
    combined_prompt = (
        f"你是一位资深的艺术评论家与美学学者，拥有对跨媒介艺术品（包括绘画、雕塑、器物等）的卓越洞察力与品味。"
        # f"你的核心任务是对这件艺术品（已知信息：名称《{artwork_info['名称']}》，作者/出处: {artwork_info['作者']}）进行一次深刻的审美评价，撰写一篇约1000字的艺术评论。"
        f"你的核心任务是对眼前这件【未知来源的匿名艺术品】进行一次纯粹基于视觉的深刻审美评价，撰写一篇约1000字的艺术评论。" # <-- 修改点
        "评论的重点应是其艺术魅力和审美价值，而非纯粹的功能或考古学分析。请使用Markdown格式，并围绕以下几个核心美学维度展开，确保论述充满洞见，并与作品的具体视觉元素紧密结合：\n\n"
        "### 1. 审美风格与艺术创新 (Aesthetic Style and Artistic Innovation)\n"
        "   - 将该作品置于其所属的艺术风格或审美潮流中进行审视。它在多大程度上体现、发展或挑战了当时的审美理想？\n"
        "   - 评价其在美学上的独创性。这件作品的美，是一种对既有范式的完美复现，还是一种开创性的、定义了新审美方向的探索？其独特的艺术价值在何处？\n\n"
        "### 2. 造型与构成的美学 (The Aesthetics of Form and Composition)\n"
        "   - 在上述风格框架下，具体分析作品的整体形态、结构和比例所创造出的美感。是和谐、平衡，还是充满张力与动感？\n"
        "   - 解读其线条的运用、轮廓的起伏、或形态的组合，是如何引导观者的视线并营造出独特的视觉韵律的。\n\n"
        "### 3. 材质、工艺与细节之美 (The Beauty of Material, Craftsmanship, and Detail)\n"
        "   - 欣赏并评述艺术家对材质（如墨与纸、石料的纹理、釉面的光泽、青铜的质感）的独特处理及其所呈现的审美效果。\n"
        "   - 深入分析其制作工艺（如笔法、刀法、塑形、上釉）的精湛之处。这些技艺是如何服务于最终的审美表达，而不仅仅是技术本身？细微之处（如一笔顿挫、一处雕痕）如何体现了艺术家的巧思与品味？\n\n"
        "### 4. 情感、氛围与意境 (Emotion, Atmosphere, and Artistic Mood)\n"
        "   - 综合以上分析，深入探讨这件作品所传达的核心情感与营造的整体氛围。是宁静致远、雄伟壮丽，还是内敛含蓄、华贵典雅？\n"
        "   - 最终阐释作品是如何通过其独特的风格、构成与工艺，共同作用创造出一种超越物象本身的“意境”或艺术感染力的。这是作品的灵魂所在。\n"
    )
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": [{"type": "text", "text": combined_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]        )
        return {
            "model_name": model_key,
            "response": response.choices[0].message.content,
            "model_info": { "name": model_name, "provider": model_details.get("provider", "Unknown") }
        }
    except Exception as e:
        return {"error": f"模型 {model_name} 在处理时发生错误: {str(e)}"}
# ==============================================================================
# 页面渲染路由 (Page Routes)
# ==============================================================================
@app.route('/')
def gallery_page():
    if datas is None:
        return "数据文件未加载，无法显示画廊。", 500

    # --- 新增：定义筛选分类和获取当前选择 ---
    era_groups = ['唐前', '宋元', '明', '清', '近现代']
    selected_era = request.args.get('era', None)

    # --- 新增：根据选择筛选数据 ---
    if selected_era and selected_era in era_groups:
        # 如果有合法的筛选条件，则按条件筛选
        filtered_data = datas[datas['era_group'] == selected_era]
        # 为防止卡顿，即使筛选后也只显示部分数据，比如最多100条
        artworks_to_display = filtered_data.sample(10).to_dict('records')
    else:
        # 如果没有筛选条件（即显示“全部”），则随机展示20条
        artworks_to_display = datas.sample(n=min(len(datas), 10)).to_dict('records')

    return render_template('gallery.html', 
                           artworks=artworks_to_display,
                           era_groups=era_groups,        # 把所有分类传给前端
                           selected_era=selected_era)    # 把当前选中的分类传给前端


@app.route('/artwork/<artwork_id>')
def artwork_detail_page(artwork_id):
    if datas is None:
        return "数据文件未加载。", 500
    artwork_data = datas[datas['id'] == artwork_id]
    if artwork_data.empty:
        abort(404)
    return render_template('artwork_detail.html', artwork=artwork_data.iloc[0].to_dict())

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIRECTORY, filename)

# ==============================================================================
# API 接口路由 (API Routes)
# ==============================================================================

@app.route('/api/artwork/evaluate', methods=['POST'])
def evaluate_artwork_api():
    data = request.get_json()
    if not data or 'artwork_id' not in data:
        return jsonify({"error": "请求体必须包含 'artwork_id'"}), 400
    artwork_id = data['artwork_id']
    artwork_info = datas[datas['id'] == artwork_id]
    if artwork_info.empty:
        return jsonify({"error": f"ID为 '{artwork_id}' 的艺术品未找到"}), 404
    available_models = list(MODEL_CONFIG.keys())
    if len(available_models) < 2:
        return jsonify({"error": "配置的模型少于2个，无法进行比较"}), 500
    model_keys = model_sampler.select_pair()
    print(f"🔄 [服务端] 随机选择模型: {model_keys} 对作品《{artwork_info.iloc[0]['名称']}》进行评价")
    evaluations = {}
    for key in model_keys:
        evaluations[key] = run_art_cot_analysis(key, artwork_info.iloc[0])
    return jsonify({"evaluations": evaluations})


@app.route('/api/artwork/evaluate_anonymous', methods=['POST'])
def evaluate_artwork_anonymous_api():
    """匿名评价接口，不提供作品元数据"""
    data = request.get_json()
    if not data or 'artwork_id' not in data:
        return jsonify({"error": "请求体必须包含 'artwork_id'"}), 400
    artwork_id = data['artwork_id']
    artwork_info = datas[datas['id'] == artwork_id]
    if artwork_info.empty:
        return jsonify({"error": f"ID为 '{artwork_id}' 的艺术品未找到"}), 404
    
    available_models = list(MODEL_CONFIG.keys())
    if len(available_models) < 2:
        return jsonify({"error": "配置的模型少于2个，无法进行比较"}), 500
    
    model_keys = random.sample(available_models, 2)
    # 在匿名模式下，我们只打印ID，不泄露名称
    print(f"🔄 [服务端] 随机选择模型: {model_keys} 对作品 ID: {artwork_id} 进行【匿名】评价")
    
    evaluations = {}
    for key in model_keys:
        # 调用新增的匿名分析函数
        evaluations[key] = run_art_cot_analysis_anonymous(key, artwork_info.iloc[0])
        
    return jsonify({"evaluations": evaluations})



@app.route('/api/evaluation/save', methods=['POST'])
def save_evaluation_api():
    evaluation_id = str(uuid.uuid4())
    return jsonify({"evaluation_id": evaluation_id})

@app.route('/api/vote', methods=['POST'])
def vote_api():
    data = request.get_json()
    required_fields = ['evaluation_id', 'winner', 'artwork_id', 'artwork_name', 'model_a', 'model_b', 'response_a', 'response_b']
    if not data or not all(field in data for field in required_fields):
        return jsonify({"error": "请求体缺少必要字段"}), 400
    rating_record = {
        'timestamp': datetime.now().isoformat(), 'evaluation_id': data['evaluation_id'],
        'artwork_id': data['artwork_id'], 'artwork_name': data['artwork_name'],
        'winner': data['winner'], 'model_a': data['model_a'], 'model_b': data['model_b'],
        'response_a': data['response_a'], 'response_b': data['response_b']
    }
    try:
        df = pd.DataFrame([rating_record])
        df.to_csv(RATINGS_FILE_PATH, mode='a', header=not os.path.exists(RATINGS_FILE_PATH), index=False)
        print(f"👍 [服务端] 收到并记录一笔新投票 (ID: {data['evaluation_id']})")
        stats = {'model_a': random.randint(5, 20), 'model_b': random.randint(5, 20), 'tie': random.randint(1, 10)}
        return jsonify({"message": "投票成功", "stats": stats})
    except Exception as e:
        print(f"❌ [服务端] 写入评分文件失败: {e}")
        return jsonify({"error": "服务器无法保存评分"}), 500
@app.route('/api/feedback', methods=['POST'])
def feedback_api():
    """V7更新: 专门接收和更新反馈到 feedback.csv"""
    data = request.get_json()
    evaluation_id = data.get('evaluation_id')
    feedback_text = data.get('feedback', '')

    if not evaluation_id:
        return jsonify({"error": "evaluation_id is required"}), 400

    feedback_record = {
        'timestamp': datetime.now().isoformat(),
        'evaluation_id': evaluation_id,
        'feedback': feedback_text
    }
    
    try:
        df = pd.DataFrame([feedback_record])
        df.to_csv(FEEDBACK_FILE_PATH, mode='a', header=not os.path.exists(FEEDBACK_FILE_PATH), index=False)
        print(f"👍 [服务端] 反馈已记录到 feedback.csv (ID: {evaluation_id})")
        return jsonify({"message": "Feedback saved successfully"})
    except Exception as e:
        print(f"❌ [服务端] 写入反馈文件 feedback.csv 失败: {e}")
        return jsonify({"error": "Failed to save feedback"}), 500
@app.route('/api/error-report', methods=['POST'])
def error_report_api():
    """处理用户错误报告"""
    data = request.get_json()
    evaluation_id = data.get('evaluation_id')
    artwork_id = data.get('artwork_id')
    
    if not evaluation_id or not artwork_id:
        return jsonify({"error": "evaluation_id and artwork_id are required"}), 400

    # 获取用户IP
    user_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
    
    error_report_record = {
        'timestamp': datetime.now().isoformat(),
        'user_ip': user_ip,
        'evaluation_id': evaluation_id,
        'artwork_id': artwork_id
    }
    
    try:
        df = pd.DataFrame([error_report_record])
        df.to_csv(ERROR_REPORT_FILE_PATH, mode='a', header=not os.path.exists(ERROR_REPORT_FILE_PATH), index=False)
        print(f"⚠️ [服务端] 错误报告已记录 (ID: {evaluation_id}, IP: {user_ip})")
        return jsonify({"message": "Error report saved successfully"})
    except Exception as e:
        print(f"❌ [服务端] 写入错误报告文件失败: {e}")
        return jsonify({"error": "Failed to save error report"}), 500
# ==============================================================================
# 服务器启动
# ==============================================================================
if __name__ == '__main__':
    print("🚀 [服务端 V6.1] 服务准备启动...")
    app.run(host='0.0.0.0', port=5022, debug=True)
