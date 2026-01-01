# FedGUI 算法实现

## 概述

FedGUI (Federated Learning with GUI-based Client Selection) 是一个基于客户端特征的自适应联邦学习算法。该算法通过分析客户端数据的特征（图像和文本）来计算客户端权重，从而实现更智能的模型聚合。

## 功能特点

- **基于特征的客户端权重计算**: 使用CLIP模型提取客户端数据的图像和文本特征
- **自适应聚合**: 根据客户端特征相似度动态调整聚合权重
- **容错机制**: 当特征计算失败时，自动回退到标准联邦平均算法
- **多模态支持**: 支持图像和文本的联合特征提取
- **模块化设计**: 代码结构清晰，易于维护和扩展

## 文件结构

```
FedMobile/swift/llm/
├── sft.py                          # 主要的联邦学习训练脚本
├── fedgui_utils/
│   └── client_features.py          # 客户端特征计算模块
├── utils/
│   └── argument.py                 # 参数定义（已添加fedgui选项）
├── test_fedgui.py                  # 测试脚本
└── fedgui_README.md                # 使用说明文档
```

## 核心函数

### 基础特征提取函数
- `get_image_vector(image_path)`: 提取图像特征向量 (512维)
- `get_text_vector(text)`: 提取文本特征向量 (512维)
- `get_joint_vector(image_path, text, mode, lam)`: 提取拼接特征向量 (1024维)
- `get_client_vector(samples, cid, mode, lam)`: 计算客户端特征向量 (1024维)

### FedGUI算法函数
- `extract_client_data_from_samples(samples)`: 从训练样本中提取客户端数据
- `compute_fedgui_weights_for_clients(online_clients, client_samples_dict, mode, lam)`: 计算FedGUI权重
- `aggregate_with_fedgui_weights(...)`: 使用FedGUI权重进行模型聚合
- `fedgui_aggregation(...)`: FedGUI聚合的主函数

### 辅助函数
- `load_json_group_by_client(json_path)`: 按客户端分组加载JSON数据
- `compute_client_weights(client_vectors)`: 计算客户端权重

## 使用方法

### 1. 基本用法

在训练命令中指定 `--fed_alg fedgui`：

```bash
python -m swift.llm.sft \
    --model_type qwen2.5-7b \
    --dataset your_dataset.json \
    --fed_alg fedgui \
    --client_num 10 \
    --round 100 \
    --output_dir ./output_fedgui
```

### 2. 参数说明

- `--fed_alg fedgui`: 使用FedGUI算法
- `--client_num`: 客户端数量
- `--round`: 训练轮数
- `--client_sample`: 每轮参与的客户端数量

### 3. 数据格式要求

数据集应包含以下字段：
```json
{
    "client_id": 0,
    "images": "/path/to/image.jpg",
    "query": "图像描述或查询文本"
}
```

## 算法流程

1. **数据准备**: 从训练样本中提取客户端数据
2. **特征提取**: 对每个客户端的数据使用CLIP模型提取图像和文本特征
3. **特征拼接**: 将图像特征(512维)和文本特征(512维)拼接成1024维向量
4. **权重计算**: 基于拼接后的特征向量计算客户端间的余弦相似度权重
5. **模型聚合**: 使用计算出的权重进行加权模型聚合
6. **容错处理**: 如果特征计算失败，回退到标准联邦平均

## 特征拼接方法

### 拼接策略
- **图像+文本模式**: 将512维图像特征和512维文本特征直接拼接为1024维向量
- **仅图像模式**: 图像特征 + 512维零向量 = 1024维向量
- **仅文本模式**: 512维零向量 + 文本特征 = 1024维向量

### 优势
- **保留完整信息**: 拼接保留了图像和文本的完整特征信息
- **更好的区分性**: 1024维向量提供更丰富的特征表示
- **统一的相似度计算**: 所有模式都使用相同维度的向量进行余弦相似度计算

## 代码简化优势

### 之前的实现
```python
elif args.fed_alg == 'fedgui':
    # 大量内联代码...
    # 复杂的特征计算逻辑
    # 冗长的聚合过程
```

### 现在的实现
```python
elif args.fed_alg == 'fedgui':
    # 准备客户端数据字典
    client_samples_dict = {}
    for j in online_clients:
        client_samples_dict[j] = get_dataset_this_round(train_dataset, i, splits[j], args)
    
    # 使用FedGUI聚合
    global_lora = fedgui_aggregation(
        global_lora=global_lora,
        local_lora_list=local_lora_list,
        online_clients=online_clients,
        client_samples_dict=client_samples_dict,
        client_num_samples=client_num_samples,
        mode="joint",
        lam=0.5
    )
```

## 模块化设计优势

1. **代码复用**: 函数可以在其他地方重复使用
2. **易于测试**: 每个函数都可以独立测试
3. **易于维护**: 修改逻辑时只需要修改相应的函数
4. **易于扩展**: 可以轻松添加新的特征提取方法或聚合策略

## 注意事项

1. **依赖要求**: 需要安装CLIP模型相关依赖
2. **内存使用**: 特征计算可能消耗较多GPU内存
3. **数据格式**: 确保数据包含正确的图像路径和查询文本
4. **性能考虑**: 特征计算可能增加训练时间

## 测试

运行测试脚本验证功能：

```bash
cd FedMobile/swift/llm
python test_fedgui.py
```

测试脚本包含：
- 基础功能测试
- 新函数测试
- 集成功能测试

## 扩展

可以通过修改以下参数来自定义算法行为：
- 特征提取模式 (`mode`)
- 图像文本融合权重 (`lam`)
- 相似度计算方法
- 权重归一化策略

## 性能优化建议

1. **特征缓存**: 可以添加特征向量缓存机制
2. **批量处理**: 可以批量处理多个客户端的特征提取
3. **并行计算**: 可以并行计算多个客户端的特征
4. **内存优化**: 可以优化内存使用，减少GPU内存占用
