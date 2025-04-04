import json
import os
from main import test

# 加载配置
config_path = 'config.json'
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# 确保模式为测试
config['mode'] = 'test'

# 运行测试
print("开始测试模型...")
test(config)
print("测试完成，请查看输出文件:", os.path.abspath(config['output_file'])) 