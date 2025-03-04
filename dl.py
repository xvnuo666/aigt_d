from huggingface_hub import snapshot_download
import os

# 配置参数
# MODEL_NAME = "roberta-base"
MODEL_NAME = "CarlanLark/AIGT-detector-mixed-source"
SAVE_DIR = "."  # 本地保存路径
HF_ENDPOINT = "https://hf-mirror.com"  # 国内镜像加速

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

try:
    # 下载完整模型库（含所有必要文件）
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=SAVE_DIR,
        resume_download=True,
        local_dir_use_symlinks=False,  # 避免符号链接
        endpoint=HF_ENDPOINT,
        ignore_patterns=["*.safetensors", "*.h5"],  # 排除非PyTorch格式文件
        max_workers=4  # 多线程下载加速
    )
    print(f"✅ 成功下载 {MODEL_NAME} 到 {SAVE_DIR}")

    # 验证关键文件是否存在
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "vocab.json",
        "merges.txt"
    ]

    missing_files = [
        f for f in required_files
        if not os.path.exists(os.path.join(SAVE_DIR, f))
    ]

    if missing_files:
        raise FileNotFoundError(f"缺失关键文件: {missing_files}")

except Exception as e:
    print(f"❌ 下载失败: {str(e)}")
    # 清理不完整文件
    if os.path.exists(SAVE_DIR):
        import shutil

        shutil.rmtree(SAVE_DIR)