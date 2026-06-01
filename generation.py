# =========================================
# SID FINAL MULTIMODAL PPO SYSTEM (PAPER)
# =========================================

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.distributions import Categorical
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
import torchvision.transforms as T
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# =========================================
# CONFIG
# =========================================
class Config:
    csv_path = "/home/One/Two/data/output_with_full_attr2023.csv"
    model_path = "/home/ubuntu/Public/QwenQwen2.5-VL-3B-Instruct"
    # model_path = "/home/ubuntu/Public/bert-base-uncased"

    state_dim = 256
    hidden_dim = 256
    num_layers = 4
    codebook_size = 256
    K=3   #close loop
    N=10000   #agent loop
    lr = 3e-4
    eps_clip = 0.2
    entropy_coef = 0.02
    lambda_u = 0.3
    
    batch_size = 64
    num_workers = 8
    seed = 42

    device = torch.device("cuda")

cfg = Config()
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
# =========================================
# DATASET（鲁棒）
# =========================================
class Dataset:
    def __init__(self):
        try:
            self.df = pd.read_csv(cfg.csv_path, encoding="utf-8")
        except:
            self.df = pd.read_csv(cfg.csv_path, encoding="gbk")

        self.df.columns = [str(c).strip().replace("\ufeff", "") for c in self.df.columns]

        def match(keys):
            for c in self.df.columns:
                for k in keys:
                    if k.lower() in c.lower():
                        return c
            return None

        self.col_id = match(["项目编号", "asin", "id"]) or self.df.columns[0]
        self.col_text = match(["description", "文本", "title"]) or self.df.columns[1]
        self.col_img = match(["image"])

        if self.col_img is None:
            self.df["__img__"] = ""
            self.col_img = "__img__"

        print("字段映射:")
        print("ID:", self.col_id)
        print("TEXT:", self.col_text)
        print("IMG:", self.col_img)

    def __len__(self):
        return len(self.df)

    def get(self, i):
        row = self.df.iloc[i]
        return {
            "asin": str(row[self.col_id]),
            "text": str(row[self.col_text]),
            "image": str(row[self.col_img])
        }

# =========================================
# ENCODER（多模态）
# =========================================
# class Encoder:
#     def __init__(self):
#         print("Loading BERT...")

#         model_path = "/home/ubuntu/Public/bert-base-uncased"

#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)

#         self.model = AutoModel.from_pretrained(model_path)
#         self.model = self.model.to(cfg.device)
#         self.model.eval()

#         self.hidden = self.model.config.hidden_size  # 768

#         self.proj = nn.Linear(self.hidden, cfg.hidden_dim).to(cfg.device)

#     @torch.no_grad()
#     def encode_text(self, text):
#         if not text or text == "nan":
#             text = "unknown item"

#         inputs = self.tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             max_length=256,
#             padding="max_length"
#         )

#         # 放到GPU
#         inputs = {k: v.to(cfg.device) for k, v in inputs.items()}

#         outputs = self.model(**inputs)

#         # mean pooling
#         x = outputs.last_hidden_state.mean(dim=1)

#         x = self.proj(x)

#         return x.squeeze(0)

#     def encode(self, item):
#         x = self.encode_text(item["text"])

#         # 强制对齐 state_dim（防止你之前的 shape error）
#         if x.shape[0] > cfg.state_dim:
#             x = x[:cfg.state_dim]
#         elif x.shape[0] < cfg.state_dim:
#             pad = cfg.state_dim - x.shape[0]
#             x = torch.nn.functional.pad(x, (0, pad))

#         return x
# =========================================
# ENCODER（多模态LLM 支持 文本 + 多张图片URL）
# =========================================

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_h = cfg.hidden_dim
        self.K = cfg.K
        self.device = cfg.device

        # Visual encoder
        self.vis_encoder = resnet18(pretrained=True)
        self.vis_encoder = nn.Sequential(*list(self.vis_encoder.children())[:-1]).to(self.device)
        self.vis_proj = nn.Linear(512, self.d_h).to(self.device)
        self.vis_encoder.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Text encoder
        self.text_proj = nn.Sequential(
            nn.Linear(768, self.d_h),
            nn.LayerNorm(self.d_h)
        ).to(self.device)

        # Structured semantic anchor encoder
        self.struct_encoder = nn.Sequential(
            nn.Linear(768, self.d_h),
            nn.LayerNorm(self.d_h)
        ).to(self.device)

        # Dual-gate
        self.Wg = nn.Linear(2 * self.d_h, self.d_h).to(self.device)
        self.Wa = nn.Linear(2 * self.d_h, self.d_h).to(self.device)

        self.out_proj = nn.Linear(2 * self.d_h, cfg.state_dim).to(self.device)

    def load_multi_images(self, img_str):
        if not img_str or img_str == "nan":
            return []
        urls = [u.strip() for u in img_str.split(",") if u.strip()]
        imgs = []
        for url in urls:
            try:
                resp = requests.get(url, timeout=2)
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                imgs.append(self.transform(img).unsqueeze(0).to(self.device))
            except:
                continue
        return imgs[:3]

    def text_embedding(self, text):
        if not text or text == "nan":
            text = "product"
        vec = np.zeros(768)
        for i, c in enumerate(text.lower()):
            vec[i % 768] += ord(c) / 200.0
        return torch.tensor(vec, dtype=torch.float32).to(self.device)

    # ----------------------------------------------------------------
    # Structured Semantic Anchor from YOUR 7 CSV columns
    # ----------------------------------------------------------------
    def get_structured_anchor(self, item):
        title = str(item.get("title", ""))
        avg_rating = str(item.get("average_rating", ""))
        rating_num = str(item.get("rating_number", ""))
        store = str(item.get("store", ""))
        categories = str(item.get("categories", ""))
        details = str(item.get("details", ""))

        structured_text = f"{title} {store} {categories} {details} {avg_rating} {rating_num}"
        feat = self.text_embedding(structured_text)
        return self.struct_encoder(feat)

    def closed_loop_fusion(self, E_tex, E_vis, E_str):
        h_vis = E_str.clone()
        h_tex = E_str.clone()

        for _ in range(self.K):
            # Suppression gate
            g_s = torch.sigmoid(self.Wg(torch.cat([h_vis, E_vis], dim=-1)))
            h_vis_p = (1 - g_s) * h_vis + g_s * E_vis
            h_vis_p = F.layer_norm(h_vis_p, (self.d_h,))

            # Anchoring gate
            g_a = torch.sigmoid(self.Wa(torch.cat([h_vis_p, E_str], dim=-1)))
            h_vis = g_s * E_vis + g_a * h_vis_p

            # Text update
            g_s_t = torch.sigmoid(self.Wg(torch.cat([h_tex, E_tex], dim=-1)))
            h_tex_p = (1 - g_s_t) * h_tex + g_s_t * E_tex
            h_tex_p = F.layer_norm(h_tex_p, (self.d_h,))

            g_a_t = torch.sigmoid(self.Wa(torch.cat([h_tex_p, E_str], dim=-1)))
            h_tex = g_s_t * E_tex + g_a_t * h_tex_p

        return torch.cat([h_tex, h_vis], dim=-1)

    @torch.no_grad()
    def encode(self, item):
        # Visual feature
        imgs = self.load_multi_images(item["image"])
        if not imgs:
            E_vis = torch.zeros(self.d_h).to(self.device)
        else:
            feats = [self.vis_encoder(img).flatten(1) for img in imgs]
            E_vis = torch.stack(feats).mean(0)
            E_vis = self.vis_proj(E_vis).squeeze(0)

        # Text feature
        E_tex = self.text_embedding(item["text"])
        E_tex = self.text_proj(E_tex)

        # Structured semantic anchor (YOUR 7 CSV FIELDS)
        E_str = self.get_structured_anchor(item)

        # Closed-loop multimodal fusion
        E_u = self.closed_loop_fusion(E_tex, E_vis, E_str)

        # Final output x
        x = self.out_proj(E_u)
        x = F.normalize(x, dim=-1)

        if x.size(0) > cfg.state_dim:
            x = x[:cfg.state_dim]
        elif x.size(0) < cfg.state_dim:
            x = F.pad(x, (0, cfg.state_dim - x.size(0)))

        return x
# =========================================
# MODEL
# =========================================
class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.codebooks = nn.ModuleList([
            nn.Embedding(cfg.codebook_size, cfg.state_dim)
            for _ in range(cfg.num_layers)
        ])

        self.actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, cfg.codebook_size)
            ) for _ in range(cfg.num_layers)
        ])

        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ) for _ in range(cfg.num_layers)
        ])

# =========================================
# TRAINER（核心）
# =========================================
class Trainer:
    def __init__(self, encoder, model):
        self.encoder = encoder
        self.model = model.to(cfg.device)
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.global_usage = [
            torch.zeros(cfg.codebook_size).to(cfg.device)
            for _ in range(cfg.num_layers)
        ]
        self.usage = [
            torch.zeros(cfg.codebook_size).to(cfg.device)
            for _ in range(cfg.num_layers)
        ]

    def get_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def train_step(self, item, temperature):

        e = self.encoder.encode(item)

        m = self.get_model()

        states, actions, logps, values, logits_all = [], [], [], [], []
        state = e

        # ===== forward =====
        for l in range(cfg.num_layers):
            logits = m.actors[l](state)

            dist = Categorical(logits=logits / temperature)

            action = dist.sample()
            self.global_usage[l][action] += 1
            
            logp = dist.log_prob(action)
            value = m.critics[l](state)

            states.append(state)
            actions.append(action)
            logps.append(logp)
            values.append(value)
            logits_all.append(logits)

            code = m.codebooks[l](action)

            state = state - code
            state = state / (state.norm() + 1e-6)

            self.usage[l][action] += 1

        # # ===== reward =====
        # rewards = []
        # recon = torch.zeros_like(e)

        # for l in reversed(range(cfg.num_layers)):
        #     code = m.codebooks[l](actions[l])
        #     recon += code

        #     rec_loss = torch.norm(e - recon) ** 2

        #     p = torch.softmax(logits_all[l], dim=-1)
        #     uniform = torch.ones_like(p) / cfg.codebook_size
        #     kl = (p * (torch.log(p + 1e-8) - torch.log(uniform))).sum()

        #     usage_penalty = torch.log(self.usage[l][actions[l]] + 1)

        #     r = -(rec_loss + cfg.lambda_u * kl + 0.2 * usage_penalty)
        #     rewards.insert(0, r)

        # ===== 多层面分层奖励 =====
        rewards = []
        residual = e.clone()  # 初始残差 = 原始特征

        # 正序遍历每一层，逐层计算残差与分层奖励
        for l in range(cfg.num_layers):
            code = m.codebooks[l](actions[l])
            
            # 1. 逐层残差重构：当前层只拟合当前残差（RQ-VAE 核心）
            rec_loss = torch.norm(residual - code) ** 2
            
            # 2. 分层KL均匀约束：当前层策略分布逼近均匀分布
            p = torch.softmax(logits_all[l], dim=-1)
            uniform = torch.ones_like(p) / cfg.codebook_size
            kl = (p * (torch.log(p + 1e-8) - torch.log(uniform))).sum()

            # 3. 分层码本使用惩罚：约束当前层编码使用频率
            usage_penalty = torch.log(self.usage[l][actions[l]] + 1)

            # 当前层专属多层面奖励
            layer_reward = -(rec_loss + cfg.lambda_u * kl + 0.2 * usage_penalty)
            rewards.append(layer_reward)

            # 更新残差：减去当前层编码，传递给下一层
            residual = residual - code
            residual = residual / (residual.norm() + 1e-6)

        # ===== PPO =====
        loss = 0
        for l in range(cfg.num_layers):
            dist = Categorical(logits=m.actors[l](states[l]))

            new_logp = dist.log_prob(actions[l])
            ratio = torch.exp(new_logp - logps[l].detach())

            adv = rewards[l] - values[l].detach()

            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - cfg.eps_clip, 1 + cfg.eps_clip) * adv

            actor = -torch.min(s1, s2)
            critic = (values[l] - rewards[l]) ** 2
            entropy = dist.entropy()

            loss += actor + critic - cfg.entropy_coef * entropy

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item(), actions

# =========================================
# TRAIN
# =========================================
def train():
    dataset = Dataset()
    encoder = Encoder()
    model = Quantizer()

    # ===== 多GPU =====
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    trainer = Trainer(encoder, model)

    os.makedirs("outputs2", exist_ok=True)

    # ===== 全局SID（可选累计）=====
    global_sid = {}

    for epoch in range(cfg.N):
        print(f"\n========== Epoch {epoch} ==========")

        epoch_sid = {}
        usage = [set() for _ in range(cfg.num_layers)]

        # 温度退火（关键，防止塌缩）
        temperature = max(0.5, 2.0 - epoch * 0.3)

        for i in tqdm(range(len(dataset))):
            item = dataset.get(i)

            loss, acts = trainer.train_step(item, temperature)

            asin = item["asin"]
            # act_list = [a.item() for a in acts]
            act_list = [
                f"<a_{acts[0].item()}>",
                f"<b_{acts[1].item()}>",
                f"<c_{acts[2].item()}>",
                f"<d_{acts[3].item()}>"
            ]
            # ===== 保存SID =====
            epoch_sid[asin] = act_list
            global_sid[asin] = act_list

            # ===== usage统计 =====
            for l in range(cfg.num_layers):
                usage[l].add(act_list[l])

        # ===============================
        # 📊 打印 usage（论文核心指标）
        # ===============================
        print("\n📊 Codebook Usage:")
        for l in range(cfg.num_layers):
            print(f"Layer {l}: {len(usage[l])}/{cfg.codebook_size}")

        # ===============================
        # 💾 每个epoch保存
        # ===============================

        # 1️⃣ SID
        with open(f"outputs2/sid_epoch_{epoch}.json", "w") as f:
            json.dump(epoch_sid, f, indent=2)

        # 2️⃣ 全局SID（可选）
        with open("outputs2/sid_all.json", "w") as f:
            json.dump(global_sid, f)

        # 3️⃣ Codebook embedding
        for l in range(cfg.num_layers):
            emb = (
                model.module.codebooks[l].weight
                if isinstance(model, nn.DataParallel)
                else model.codebooks[l].weight
            )
            np.save(
                f"outputs2/codebook_epoch{epoch}_L{l}.npy",
                emb.detach().cpu().numpy()
            )

        # 4️⃣ usage向量（更细粒度分析）
        for l in range(cfg.num_layers):
            usage_tensor = trainer.global_usage[l].detach().cpu().numpy()
            np.save(f"outputs2/usage_epoch{epoch}_L{l}.npy", usage_tensor)

        print(f"✅ Epoch {epoch} saved")

    print("\n🎉 Training Finished")
# =========================================
if __name__ == "__main__":
    train()
