这是一个为您的 Streamlit 本地聊天应用量身定制的 GitHub README 文件。

---

# LocalChat: Streamlit 本地 GGUF 模型聊天应用

这是一个基于 Python、[Streamlit](https://streamlit.io/) 和 [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) 实现的本地聊天机器人应用。它允许你在本地机器上运行 **GGUF** 格式的语言模型，并提供一个友好的 Web 界面，支持聊天历史和流式输出。

## ✨ 特性

*   **本地模型支持：** 使用高效的 `llama-cpp-python` 库运行 GGUF 格式的模型。
*   **Streamlit Web 界面：** 简洁、易用的聊天界面。
*   **聊天历史：** 通过 Streamlit 的 `session_state` 维护完整的对话历史。
*   **流式输出：** 模型回答以流式（打字机）效果显示，提升用户体验。
*   **遵循对话模板：** 使用 `llm.create_chat_completion` 确保模型以正确的对话格式进行推理。
*   **GPU 加速：** 支持通过 `n_gpu_layers` 参数使用 GPU 加速（需正确安装 `llama-cpp-python`）。

## 🛠️ 环境准备

### 1. 克隆项目

```bash
git clone <YOUR_REPO_URL>
cd localchat
```

### 2. Python 依赖安装

您需要根据您的硬件环境选择安装 `llama-cpp-python`。

#### 📦 CPU 模式 (推荐配置)

如果您没有独立显卡或不打算使用 GPU 加速：

```bash
pip install streamlit llama-cpp-python
```

#### 🚀 GPU 加速模式 (CUDA)

如果您有 NVIDIA 显卡并希望使用 GPU 加速（**强烈推荐性能更好的方式**）：

```bash
# 请确保您的系统已安装 CUDA Toolkit
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" 启用 CUDA 支持
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
pip install streamlit
```

> **注意：** 更多高级安装选项（如 Metal/Apple Silicon）请参考 [llama-cpp-python 官方文档](https://github.com/abetlen/llama-cpp-python#installation)。

## ⚙️ 配置与使用

### 1. 下载 GGUF 模型

您需要从 Hugging Face 等平台下载一个 **GGUF** 格式的模型文件。

例如，您可以前往 [Hugging Face GGUF 模型集合](https://huggingface.co/models?pipeline_tag=text-generation&library=llama-cpp-python&sort=trending) 下载适合您的模型。

将下载好的模型文件放置到您项目的某个路径下。

### 2. 修改 `app.py` 配置

打开 `app.py` 文件，修改以下两个核心配置参数：

```python
# --- 配置 ---
# 替换为您的模型文件的实际路径和文件名
MODEL_PATH =    "xxx.gguf"             # <--- 将 "xxx.gguf" 替换为您的模型路径和文件名
# 根据您的硬件调整 n_gpu_layers，用于GPU加速。
# 如果没有独立显卡或显存不足，请设置为 0。
N_GPU_LAYERS = 0 # 例如：设置为30来使用GPU加速
```

*   将 `MODEL_PATH` 更改为您下载的 GGUF 文件的路径。
*   如果您安装了 GPU 版本并希望启用加速，请将 `N_GPU_LAYERS` 设置为一个大于 0 的整数（通常是模型总层数的大部分，可以从模型页面或实验中获得）。如果使用 CPU，请保持为 `0`。

### 3. 运行应用

在终端中运行以下命令启动 Streamlit 应用：

```bash
streamlit run app.py
```

应用将自动在您的浏览器中打开（通常是 `http://localhost:8501`），模型会在第一次运行时加载。

## 核心代码概览

应用的核心逻辑在于 `load_llm` 和 `generate_response` 函数：

### 模型加载

使用 `@st.cache_resource` 缓存模型对象，确保在 Streamlit 重新运行时模型不会被重复加载。

```python
@st.cache_resource
def load_llm():
    llm = Llama(
        model_path= MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=4096,  # 上下文窗口大小
        verbose=False,
    )
    return llm
```

### 对话生成

使用 `llm.create_chat_completion` 传入完整的 `messages` 历史记录，确保模型能够理解上下文并遵循正确的对话格式。同时，启用 `stream=True` 实现流式输出。

```python
def generate_response(messages):
    output = llm.create_chat_completion(
        messages=messages,  # 直接传入完整的历史记录
        max_tokens=2048,
        stream=True  # 启用流式输出
    )
    # ... 处理流式输出并 yield 文本片段
```
