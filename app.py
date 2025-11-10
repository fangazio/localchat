import streamlit as st
from llama_cpp import Llama
import os

# --- 配置 ---
# 替换为您的模型文件的实际路径和文件名
MODEL_PATH =    "xxx.gguf"             #路径为r"path.gguf"
# 根据您的硬件调整 n_gpu_layers，用于GPU加速。
# 如果没有独立显卡或显存不足，请设置为 0。
N_GPU_LAYERS = 0 # 例如：设置为30来使用GPU加速

# --- Streamlit 界面 ---
st.title("LocalChat")

# 检查模型文件是否存在
if not os.path.exists(MODEL_PATH):
    st.error(f"找不到模型文件：{MODEL_PATH}。请检查路径是否正确。")
    st.stop()

# 缓存模型加载，避免每次交互都重新加载
@st.cache_resource
def load_llm():
    """加载本地的 GGUF 模型"""
    with st.spinner("正在加载模型，请稍候..."):
        llm = Llama(
            model_path= MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=4096,  # 上下文窗口大小
            verbose=False,
        )
    return llm

# 加载模型
llm = load_llm()

# --- 核心问答逻辑 ---
# 修改 generate_response 函数，接受整个消息历史作为输入
def generate_response(messages):
    """调用模型生成回答，使用 create_chat_completion 遵循模型的对话模板"""

    # messages 已经是符合 create_chat_completion 要求的格式
    
    # 使用 LLM.create_chat_completion 进行推理，它会自动处理对话模板
    output = llm.create_chat_completion(
        messages=messages,  # 直接传入完整的历史记录
        max_tokens=2048,  # 最大生成 token 数
        # 移除 'stop' 参数，让 create_chat_completion 自动处理模型的停止标记
        stream=True  # 启用流式输出
    )
    
    # 流式输出结果
    full_response = ""
    # 注意：create_chat_completion 的流式输出结构与 llm() 不同
    for chunk in output:
        # 提取文本片段
        # 结构: {"choices": [{"index": 0, "delta": {"content": "..."}}]}
        text = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if text:
            full_response += text
            yield text  # 将生成的文本片段返回给Streamlit

# 初始化 Streamlit 会话状态中的聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 聊天输入框
if prompt := st.chat_input("输入你的问题..."):
    # 1. 用户消息添加到历史记录并显示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 模型回答并显示
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # *** 关键变化：传入完整的 st.session_state.messages ***
        # 调用生成函数，并流式更新显示
        for chunk in generate_response(st.session_state.messages):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌") # 模拟打字光标
        
        message_placeholder.markdown(full_response) # 最终显示完整回答

    # 3. 将模型回答添加到历史记录
    st.session_state.messages.append({"role": "assistant", "content": full_response})