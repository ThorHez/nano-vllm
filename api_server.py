#!/usr/bin/env python3
"""
OpenAI兼容的API服务器

提供类似OpenAI API的HTTP接口
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from flask import Flask, request, jsonify, Response
import json
from nanovllm import LLM, create_chat_completion
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量
llm = None
chat_completion = None


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    聊天完成接口，兼容OpenAI Chat API
    """
    try:
        data = request.get_json()
        
        # 验证必需参数
        if 'messages' not in data:
            return jsonify({'error': 'messages field is required'}), 400
        
        messages = data['messages']
        model = data.get('model', 'nanovllm')
        temperature = data.get('temperature', 0.8)
        max_tokens = data.get('max_tokens', 200)
        tools = data.get('tools')
        tool_choice = data.get('tool_choice', 'auto')
        stream = data.get('stream', False)
        
        logger.info(f"Chat completion request: {len(messages)} messages, stream={stream}")
        
        # 调用ChatCompletion
        if stream:
            return Response(
                stream_chat_completions(messages, model, temperature, max_tokens, tools, tool_choice),
                mimetype='text/plain'
            )
        else:
            response = chat_completion.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                stream=False
            )
            return jsonify(response)
            
    except Exception as e:
        logger.error(f"Error in chat completions: {e}")
        return jsonify({'error': str(e)}), 500


def stream_chat_completions(messages, model, temperature, max_tokens, tools, tool_choice):
    """流式聊天完成生成器"""
    try:
        stream = chat_completion.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            stream=True
        )
        
        for chunk in stream:
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in stream chat completions: {e}")
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"


@app.route('/v1/models', methods=['GET'])
def list_models():
    """列出可用模型"""
    return jsonify({
        "object": "list",
        "data": [{
            "id": "nanovllm",
            "object": "model",
            "created": 1677610602,
            "owned_by": "nanovllm"
        }]
    })


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({"status": "healthy"})


@app.route('/', methods=['GET'])
def index():
    """首页"""
    return jsonify({
        "message": "nanovllm OpenAI-compatible API Server",
        "endpoints": [
            "POST /v1/chat/completions - Chat completions",
            "GET /v1/models - List models",
            "GET /health - Health check"
        ]
    })


def main():
    parser = argparse.ArgumentParser(description='nanovllm API Server')
    parser.add_argument('--model', type=str, default='./models/Qwen3-0.6B',
                       help='Path to the model')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind to')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    global llm, chat_completion
    
    print("🚀 启动nanovllm API服务器")
    print(f"📚 加载模型: {args.model}")
    print(f"🖥️  设备: {args.device}")
    
    # 初始化模型
    try:
        llm = LLM(args.model, device=args.device)
        chat_completion = create_chat_completion(llm)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    print(f"🌐 服务器启动: http://{args.host}:{args.port}")
    print("📖 API文档:")
    print(f"  • POST http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  • GET  http://{args.host}:{args.port}/v1/models")
    print(f"  • GET  http://{args.host}:{args.port}/health")
    
    # 启动服务器
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main() 