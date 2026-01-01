import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

# ---- 加载 CLIP 模型 ----
device = "cuda" if torch.cuda.is_available() else "cpu"
local_model_path = "/GPFS/rhome/haotingshi/openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(local_model_path).to(device)
clip_processor = CLIPProcessor.from_pretrained(local_model_path)
clip_model.eval()


@torch.no_grad()
def get_image_vector(image_path: str):
    """单张图片 → 512 维向量"""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    feats = clip_model.get_image_features(**inputs)  # [1, 512]
    feats = F.normalize(feats, p=2, dim=-1)          # 归一化
    return feats.squeeze(0).cpu()                    # [512]

@torch.no_grad()
def get_text_vector(text: str):
    """文本 → 512 维向量"""
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
    feats = clip_model.get_text_features(**inputs)   # [1, 512]
    feats = F.normalize(feats, p=2, dim=-1)          # 归一化
    return feats.squeeze(0).cpu()                    # [512]

@torch.no_grad()
def get_joint_vector(image_path: str, text: str, mode="joint"):
    """图像+文本 → 1024 维向量（拼接）"""
    if mode == "image":
        img_vec = get_image_vector(image_path)
        # 如果没有文本，用零向量填充
        txt_vec = torch.zeros_like(img_vec)
        return torch.cat([img_vec, txt_vec], dim=0)  # [1024]
    elif mode == "text":
        txt_vec = get_text_vector(text)
        # 如果没有图像，用零向量填充
        img_vec = torch.zeros_like(txt_vec)
        return torch.cat([img_vec, txt_vec], dim=0)  # [1024]
    elif mode == "joint":
        img_vec = get_image_vector(image_path)
        txt_vec = get_text_vector(text)
        # 直接拼接图像和文本特征
        return torch.cat([img_vec, txt_vec], dim=0)  # [1024]
    else:
        raise ValueError("mode 必须是 image/text/joint")



def load_json_group_by_client(json_path):
    """解析 json 文件，按 client_id 分组，存 (image, query)"""
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    client_dict = {}
    for item in data:
        cid = item["client_id"]
        img = item["images"]
        query = item.get("query", "")   # 可能有的 json 没 query
        client_dict.setdefault(cid, []).append((img, query))
    return client_dict

def compute_client_weights(client_vectors):
    """
    输入: {cid: vector} (1024维拼接向量)
    输出: {cid: weight}，用 softmax(余弦相似度) 归一化
    """
    cids = list(client_vectors.keys())
    vecs = torch.stack([client_vectors[cid] for cid in cids])  # [N, 1024]
    mean_vec = vecs.mean(dim=0, keepdim=True)                  # [1, 1024]
    sims = F.cosine_similarity(vecs, mean_vec)                 # [N]
    alphas = torch.softmax(sims, dim=0)                        # [N]
    return {cids[i]: alphas[i].item() for i in range(len(cids))}

def compute_client_weights_with_var(client_vectors, client_vars, beta=0.2):
    """
    输入:
      - client_vectors: {cid: vector}
      - client_vars: {cid: variance}
    输出:
      - {cid: weight} (方差修正后的特征权重)
    """
    cids = list(client_vectors.keys())
    if len(cids) == 0:
        return {}
    vecs = torch.stack([client_vectors[cid] for cid in cids])  # [N, D]
    mean_vec = vecs.mean(dim=0, keepdim=True)                  # [1, D]
    sims = F.cosine_similarity(vecs, mean_vec)                 # [N]

    # 基础特征权重
    alphas = torch.softmax(sims, dim=0)                        # [N]

    # 方差修正并归一化
    adjusted = []
    for i, cid in enumerate(cids):
        var_i = float(client_vars.get(cid, 0.0))
        adjusted.append(alphas[i].item() * (1.0 + beta * var_i))
    adjusted = torch.tensor(adjusted)
    adjusted = adjusted / adjusted.sum()

    return {cids[i]: adjusted[i].item() for i in range(len(cids))}

@torch.no_grad()
def get_client_vector(samples, cid=None, mode="joint", return_var=False):
    """
    samples: [(img_path, query), ...]
    mode: "image" / "text" / "joint"
    返回:
    - 当 return_var=False 时: 1024维向量（拼接后的特征）
    - 当 return_var=True 时: (mean_vec[1024], variance[float])
    """
    vectors = []
    for img, query in tqdm(samples, desc=f"Client {cid}", ncols=80):
        try:
            if mode == "image":
                vec = get_joint_vector(img, query, mode="image")
            elif mode == "text":
                vec = get_joint_vector(img, query, mode="text")
            elif mode == "joint":
                vec = get_joint_vector(img, query, mode="joint")
            else:
                raise ValueError("mode 必须是 image/text/joint")
            vectors.append(vec)
        except Exception as e:
            print(f"⚠️ 跳过错误文件 {img}: {e}")

    if not vectors:
        raise ValueError(f"client {cid} 没有有效样本！")
    vecs = torch.stack(vectors, dim=0)  # [num_samples, 1024]
    mean_vec = vecs.mean(dim=0)
    if not return_var:
        return mean_vec
    # 使用平方差均值作为方差度量（标量）
    variance = ((vecs - mean_vec) ** 2).mean().item()
    return mean_vec, variance


# ==================== FedGUI 算法相关函数 ====================

def extract_client_data_from_samples(samples):
    """
    从训练样本中提取客户端数据
    samples: 训练样本列表
    返回: [(image_path, query), ...]
    """
    client_data = []
    for sample in samples:
        if 'images' in sample and 'query' in sample:
            client_data.append((sample['images'], sample['query']))
        elif 'image' in sample and 'query' in sample:
            client_data.append((sample['image'], sample['query']))
        else:
            # 如果没有图像和查询，使用默认值
            client_data.append(("", ""))
    return client_data


def compute_fedgui_weights_for_clients(online_clients, client_samples_dict, mode="joint"):
    """
    为在线客户端计算FedGUI权重
    
    参数:
    - online_clients: 在线客户端ID列表
    - client_samples_dict: {client_id: samples} 字典
    - mode: 特征模式 ("image", "text", "joint")
    
    返回:
    - feature_weights: {client_id: weight} 字典
    - valid_clients: 有效客户端ID列表
    """
    print("🔄 使用 FedGUI 算法进行聚合...")
    
    # 计算客户端特征向量
    client_vectors = {}
    client_vars = {}
    for j in online_clients:
        print(f"📊 计算客户端 {j} 的特征向量...")
        try:
            # 获取当前客户端的数据样本
            client_samples = client_samples_dict[j]
            
            # 将数据转换为 (image_path, query) 格式
            client_data = extract_client_data_from_samples(client_samples)
            
            # 计算客户端特征向量与方差（为了兼容旧调用，这里仅返回向量）
            if client_data:
                mean_vec, var_val = get_client_vector(client_data, cid=j, mode=mode, return_var=True)
                client_vectors[j] = mean_vec
                client_vars[j] = var_val
            else:
                print(f"⚠️ 客户端 {j} 没有有效数据，使用默认权重")
                client_vectors[j] = None
                client_vars[j] = None
                
        except Exception as e:
            print(f"⚠️ 计算客户端 {j} 特征时出错: {e}")
            client_vectors[j] = None
    
    # 过滤掉无效的客户端向量
    valid_client_vectors = {k: v for k, v in client_vectors.items() if v is not None}
    valid_client_vars = {k: client_vars[k] for k in valid_client_vectors.keys()}
    valid_clients = list(valid_client_vectors.keys())
    
    if valid_client_vectors:
        # 计算基于特征+方差修正的客户端权重（FedGUI-Var）
        feature_weights = compute_client_weights_with_var(valid_client_vectors, valid_client_vars, beta=0.2)
        print(f"📈 基于特征计算的客户端权重: {feature_weights}")
        return feature_weights, valid_clients
    else:
        print("⚠️ 所有客户端都没有有效特征")
        return {}, []


def fedgui_aggregation(global_lora, local_lora_list, online_clients, client_samples_dict, 
                      client_num_samples, mode="joint", feature_weight_alpha=0.7):
    """
    FedGUI聚合的主函数 - 支持数量权重和特征权重融合
    
    参数:
    - global_lora: 全局模型参数
    - local_lora_list: 本地模型参数列表
    - online_clients: 在线客户端ID列表
    - client_samples_dict: {client_id: samples} 字典
    - client_num_samples: 客户端数据量字典
    - mode: 特征模式 ("image", "text", "joint")
    - feature_weight_alpha: 特征权重的融合系数 (0-1), 默认0.7
    
    返回:
    - global_lora_new: 聚合后的全局模型参数
    """
    import copy
    
    print(f"🔄 使用 FedGUI 融合算法进行聚合 (α={feature_weight_alpha})...")
    
    global_lora_new = copy.deepcopy(global_lora)
    
    # 计算FedGUI权重
    feature_weights, valid_clients = compute_fedgui_weights_for_clients(
        online_clients, client_samples_dict, mode
    )
    
    # 计算数量权重 (归一化)
    total_samples = sum([client_num_samples[c] for c in online_clients])
    sample_weights = {c: client_num_samples[c] / total_samples for c in online_clients}
    
    if feature_weights:
        # 有特征权重：进行融合聚合
        print(f"📊 数量权重: {sample_weights}")
        print(f"📊 特征权重: {feature_weights}")
        print(f"📊 融合系数 α={feature_weight_alpha}")
        
        # 计算融合权重
        final_weights = {}
        for client_id in online_clients:
            if client_id in valid_clients:
                # 有特征的客户端：融合数量权重和特征权重
                # W_final = α * W_feature + (1-α) * W_sample
                final_weights[client_id] = (feature_weight_alpha * feature_weights[client_id] + 
                                          (1 - feature_weight_alpha) * sample_weights[client_id])
            else:
                # 无特征的客户端：只使用数据量权重
                final_weights[client_id] = sample_weights[client_id]
        
        # 重新归一化确保权重和为1
        total_weight = sum(final_weights.values())
        final_weights = {k: v/total_weight for k, v in final_weights.items()}
        
        print(f"📊 融合后权重: {final_weights}")
        print(f"📊 权重和验证: {sum(final_weights.values()):.6f}")
        
    else:
        # 如果所有客户端都没有有效特征，回退到标准联邦平均
        print("⚠️ 所有客户端都没有有效特征，回退到标准联邦平均")
        final_weights = sample_weights
        print(f"📊 使用数量权重: {final_weights}")
    
    # 进行加权聚合
    for net_id, client_id in enumerate(online_clients):
        net_para = local_lora_list[client_id]
        weight = final_weights[client_id]
        
        if net_id == 0:
            for key in net_para:
                global_lora_new[key] = net_para[key] * weight
        else:
            for key in net_para:
                global_lora_new[key] += net_para[key] * weight
    
    return global_lora_new


def fedgui_var_aggregation(global_lora, local_lora_list, online_clients, client_samples_dict,
                           client_num_samples, mode="joint", feature_weight_alpha=0.7, beta=0.2):
    """
    FedGUI-Var 聚合：在特征权重基础上引入客户端内部方差修正，
    最终权重: w_i = λ * α_i^{var} + (1-λ) * s_i / Σ s_j

    参数:
    - global_lora: 全局模型参数
    - local_lora_list: 本地模型参数列表 (通过 client_id 索引)
    - online_clients: 在线客户端ID列表
    - client_samples_dict: {client_id: samples} 字典
    - client_num_samples: {client_id: num_samples} 字典
    - mode: 特征模式 ("image", "text", "joint")
    - feature_weight_alpha: λ, 融合系数
    - beta: 方差修正系数
    """
    import copy

    print(f"🔄 使用 FedGUI-Var 融合算法进行聚合 (α={feature_weight_alpha}, β={beta})...")

    global_lora_new = copy.deepcopy(global_lora)

    # 复用带方差修正的权重计算流程
    feature_weights, valid_clients = compute_fedgui_weights_for_clients(
        online_clients, client_samples_dict, mode
    )

    # 计算数量权重 (归一化)
    total_samples = sum([client_num_samples[c] for c in online_clients])
    sample_weights = {c: client_num_samples[c] / total_samples for c in online_clients}

    if feature_weights:
        print(f"📊 数量权重: {sample_weights}")
        print(f"📊 特征权重(含方差修正): {feature_weights}")
        print(f"📊 融合系数 α={feature_weight_alpha}")
        print(f"📊 beta={beta}")

        final_weights = {}
        for client_id in online_clients:
            if client_id in valid_clients:
                final_weights[client_id] = (
                    feature_weight_alpha * feature_weights[client_id]
                    + (1 - feature_weight_alpha) * sample_weights[client_id]
                )
            else:
                final_weights[client_id] = sample_weights[client_id]

        total_weight = sum(final_weights.values())
        final_weights = {k: v / total_weight for k, v in final_weights.items()}
        print(f"📊 融合后权重: {final_weights}")
        print(f"📊 权重和验证: {sum(final_weights.values()):.6f}")
    else:
        print("⚠️ 所有客户端都没有有效特征，回退到标准联邦平均")
        final_weights = sample_weights
        print(f"📊 使用数量权重: {final_weights}")

    # 进行加权聚合
    for net_id, client_id in enumerate(online_clients):
        net_para = local_lora_list[client_id]
        weight = final_weights[client_id]

        if net_id == 0:
            for key in net_para:
                global_lora_new[key] = net_para[key] * weight
        else:
            for key in net_para:
                global_lora_new[key] += net_para[key] * weight

    return global_lora_new


if __name__ == "__main__":
    json_path = "/GPFS/rhome/haotingshi/datasets/test/GUI_web_test.json"  # <-- 注意是 .json
    client_dict = load_json_group_by_client(json_path)

    client_vectors = {}
    for cid, imgs in client_dict.items():
        print(f"\n👉 正在计算 client {cid} 的特征 (共 {len(imgs)} 张图)")
        client_vectors[cid] = get_client_vector(imgs, cid=cid)

    weights = compute_client_weights(client_vectors)
    print("\n📊 Client 权重：", weights)
