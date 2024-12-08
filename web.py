from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# 创建文本生成管道，使用 Qwen/Qwen2.5-3B-Instruct 模型
try:
    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct")
except Exception as e:
    print(f"模型加载失败: {e}")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    user_input = data.get('input', '')
    
    # 生成模型输出，调整参数
    try:
        response = pipe(
            user_input,
            max_new_tokens=100,  # 使用 max_new_tokens 控制生成长度
            num_return_sequences=1,
            temperature=0.5,  # 调整温度
            truncation=True,
            do_sample=True  # 启用采样
        )
        return jsonify({'response': response[0]['generated_text']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # 返回错误信息

if __name__ == '__main__':
    app.run(port=5000)