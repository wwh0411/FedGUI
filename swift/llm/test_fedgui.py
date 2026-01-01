#!/usr/bin/env python3
"""
测试 FedGUI 算法实现的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fedgui_utils.client_features import (
    load_json_group_by_client, 
    get_client_vector, 
    compute_client_weights,
    extract_client_data_from_samples,
    compute_fedgui_weights_for_clients,
    aggregate_with_fedgui_weights,
    fedgui_aggregation
)

def test_client_features():
    """测试客户端特征计算功能"""
    print("🧪 测试客户端特征计算功能...")
    
    # 创建测试数据
    test_data = [
        {
            "client_id": 0,
            "images": "/path/to/image1.jpg",
            "query": "这是一个测试查询1"
        },
        {
            "client_id": 0,
            "images": "/path/to/image2.jpg", 
            "query": "这是一个测试查询2"
        },
        {
            "client_id": 1,
            "images": "/path/to/image3.jpg",
            "query": "这是另一个测试查询"
        }
    ]
    
    # 测试按客户端分组
    client_dict = {}
    for item in test_data:
        cid = item["client_id"]
        img = item["images"]
        query = item.get("query", "")
        client_dict.setdefault(cid, []).append((img, query))
    
    print(f"📊 客户端分组结果: {client_dict}")
    
    # 测试权重计算（使用模拟数据）
    import torch
    client_vectors = {
        0: torch.randn(1024),  # 1024维拼接向量
        1: torch.randn(1024)   # 1024维拼接向量
    }
    
    weights = compute_client_weights(client_vectors)
    print(f"📈 计算出的权重: {weights}")
    
    print("✅ 基础功能测试完成！")


def test_new_functions():
    """测试新添加的FedGUI函数"""
    print("\n🧪 测试新添加的FedGUI函数...")
    
    # 测试 extract_client_data_from_samples
    test_samples = [
        {"images": "/path/to/img1.jpg", "query": "查询1"},
        {"image": "/path/to/img2.jpg", "query": "查询2"},
        {"other_field": "value"}  # 没有图像和查询的样本
    ]
    
    client_data = extract_client_data_from_samples(test_samples)
    print(f"📊 提取的客户端数据: {client_data}")
    
    # 测试 compute_fedgui_weights_for_clients（使用模拟数据）
    online_clients = [0, 1]
    client_samples_dict = {
        0: [{"images": "/path/to/img1.jpg", "query": "查询1"}],
        1: [{"images": "/path/to/img2.jpg", "query": "查询2"}]
    }
    
    # 由于没有真实的图像文件，这里会失败，但我们测试函数结构
    try:
        feature_weights, valid_clients = compute_fedgui_weights_for_clients(
            online_clients, client_samples_dict, mode="joint", lam=0.5
        )
        print(f"📈 特征权重: {feature_weights}")
        print(f"📊 有效客户端: {valid_clients}")
    except Exception as e:
        print(f"⚠️ 预期错误（因为没有真实图像文件）: {e}")
    
    # 测试聚合函数（使用模拟数据）
    import torch
    import copy
    
    # 模拟模型参数
    global_lora = {"layer1": torch.randn(10, 10), "layer2": torch.randn(5, 5)}
    local_lora_list = [
        {"layer1": torch.randn(10, 10), "layer2": torch.randn(5, 5)},
        {"layer1": torch.randn(10, 10), "layer2": torch.randn(5, 5)}
    ]
    feature_weights = {0: 0.6, 1: 0.4}
    valid_clients = [0, 1]
    client_num_samples = [100, 50]
    
    try:
        aggregated_model = aggregate_with_fedgui_weights(
            global_lora, local_lora_list, online_clients,
            feature_weights, valid_clients, client_num_samples
        )
        print("✅ 聚合函数测试成功")
    except Exception as e:
        print(f"❌ 聚合函数测试失败: {e}")
    
    print("✅ 新函数测试完成！")


def test_fedgui_integration():
    """测试FedGUI集成功能"""
    print("\n🧪 测试FedGUI集成功能...")
    
    # 模拟完整的FedGUI聚合过程
    import torch
    import copy
    
    # 模拟数据
    global_lora = {"layer1": torch.randn(10, 10), "layer2": torch.randn(5, 5)}
    local_lora_list = [
        {"layer1": torch.randn(10, 10), "layer2": torch.randn(5, 5)},
        {"layer1": torch.randn(10, 10), "layer2": torch.randn(5, 5)}
    ]
    online_clients = [0, 1]
    client_samples_dict = {
        0: [{"images": "/path/to/img1.jpg", "query": "查询1"}],
        1: [{"images": "/path/to/img2.jpg", "query": "查询2"}]
    }
    client_num_samples = [100, 50]
    
    try:
        # 测试完整的FedGUI聚合（会回退到标准聚合）
        result = fedgui_aggregation(
            global_lora=global_lora,
            local_lora_list=local_lora_list,
            online_clients=online_clients,
            client_samples_dict=client_samples_dict,
            client_num_samples=client_num_samples,
            mode="joint",
            lam=0.5
        )
        print("✅ FedGUI集成测试成功")
    except Exception as e:
        print(f"❌ FedGUI集成测试失败: {e}")
    
    print("✅ 集成测试完成！")


if __name__ == "__main__":
    test_client_features()
    test_new_functions()
    test_fedgui_integration()
    print("\n🎉 所有测试完成！")
