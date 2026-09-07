# model/ — qwen3_5 架构定义

config.json（full_attention_interval=4：64 层中 16 全注意力 + 48 GDN 线性注意力；
mtp_num_hidden_layers=1；词表 248,320）+ chat_template.jinja + generation_config.json。
权重本体不入库（来源见根 README）；量化/打包后的派生目录结构见 docs/VLLM-OPTIMIZATION.md §2。
