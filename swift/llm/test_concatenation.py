#!/usr/bin/env python3
"""
测试特征拼接方法的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

def test_concatenation_method():
    """测试特征拼接方法"""
    print("🧪 测试特征拼接方法...")
    
    # 模拟512维的图像和文本特征
    img_feat = torch.randn(512)
    txt_feat = torch.randn(512)
    
    print(f"📊 图像特征维度: {img_feat.shape}")
    print(f"📊 文本特征维度: {txt_feat.shape}")
    
    # 测试拼接
    joint_feat = torch.cat([img_feat, txt_feat], dim=0)
    print(f"📊 拼接后特征维度: {joint_feat.shape}")
    
    # 验证拼接结果
    assert joint_feat.shape[0] == 1024, f"拼接后维度应该是1024，实际是{joint_feat.shape[0]}"
    assert torch.allclose(joint_feat[:512], img_feat), "前512维应该等于图像特征"
    assert torch.allclose(joint_feat[512:], txt_feat), "后512维应该等于文本特征"
    
    print("✅ 拼接方法测试通过！")


def test_zero_padding():
    """测试零向量填充"""
    print("\n🧪 测试零向量填充...")
    
    # 模拟只有图像特征的情况
    img_feat = torch.randn(512)
    zero_vec = torch.zeros_like(img_feat)
    
    # 拼接图像特征和零向量
    img_only_feat = torch.cat([img_feat, zero_vec], dim=0)
    print(f"📊 仅图像特征拼接后维度: {img_only_feat.shape}")
    
    # 验证结果
    assert img_only_feat.shape[0] == 1024, "维度应该是1024"
    assert torch.allclose(img_only_feat[:512], img_feat), "前512维应该等于图像特征"
    assert torch.allclose(img_only_feat[512:], zero_vec), "后512维应该等于零向量"
    
    print("✅ 零向量填充测试通过！")


def test_cosine_similarity():
    """测试余弦相似度计算"""
    print("\n🧪 测试余弦相似度计算...")
    
    # 创建多个1024维的客户端特征向量
    client_vectors = {
        0: torch.randn(1024),
        1: torch.randn(1024),
        2: torch.randn(1024)
    }
    
    # 计算余弦相似度权重
    cids = list(client_vectors.keys())
    vecs = torch.stack([client_vectors[cid] for cid in cids])  # [N, 1024]
    mean_vec = vecs.mean(dim=0, keepdim=True)                  # [1, 1024]
    sims = F.cosine_similarity(vecs, mean_vec)                 # [N]
    alphas = torch.softmax(sims, dim=0)                        # [N]
    
    weights = {cids[i]: alphas[i].item() for i in range(len(cids))}
    
    print(f"📊 客户端特征向量维度: {vecs.shape}")
    print(f"📊 平均向量维度: {mean_vec.shape}")
    print(f"📊 余弦相似度: {sims}")
    print(f"📊 计算出的权重: {weights}")
    
    # 验证权重和为1
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 1e-6, f"权重和应该为1，实际为{total_weight}"
    
    print("✅ 余弦相似度计算测试通过！")


def test_different_modes():
    """测试不同模式的特征拼接"""
    print("\n🧪 测试不同模式的特征拼接...")
    
    # 模拟特征向量
    img_feat = torch.randn(512)
    txt_feat = torch.randn(512)
    
    # 测试不同模式
    modes = ["image", "text", "joint"]
    
    for mode in modes:
        if mode == "image":
            # 图像模式：图像特征 + 零向量
            joint_feat = torch.cat([img_feat, torch.zeros_like(img_feat)], dim=0)
        elif mode == "text":
            # 文本模式：零向量 + 文本特征
            joint_feat = torch.cat([torch.zeros_like(txt_feat), txt_feat], dim=0)
        elif mode == "joint":
            # 联合模式：图像特征 + 文本特征
            joint_feat = torch.cat([img_feat, txt_feat], dim=0)
        
        print(f"📊 {mode}模式特征维度: {joint_feat.shape}")
        assert joint_feat.shape[0] == 1024, f"{mode}模式维度应该是1024"
    
    print("✅ 不同模式测试通过！")


if __name__ == "__main__":
    test_concatenation_method()
    test_zero_padding()
    test_cosine_similarity()
    test_different_modes()
    print("\n🎉 所有拼接方法测试完成！")


