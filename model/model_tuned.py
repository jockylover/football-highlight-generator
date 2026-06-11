import torch
import clip
from torch import nn
import torch.nn.functional as F


class ImprovedCLIPShotDetector(nn.Module):
    def __init__(self, device="cuda", freeze_vision=False, freeze_text=False):
        super().__init__()
        self.device = device
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        # CLIP 在 CUDA 上默认以 fp16 加载，会与 fp32 的 fusion_layer 不匹配；统一转 fp32
        self.clip_model = self.clip_model.float()

        # 定义射门相关的文本描述
        self.shot_descriptions = [
            "a soccer player shooting the ball towards goal",
            "a football player kicking ball at goal",
            "soccer shot on goal",
            "football striker shooting",
            "player attempting goal",
            "soccer ball flying towards goalpost",
            "goalkeeper facing incoming shot",
            "powerful soccer shot",
            "close range goal attempt",
            "long distance football shot"
        ]

        self.non_shot_descriptions = [
            "soccer players passing the ball",
            "football players running on field",
            "goalkeeper holding the ball",
            "soccer player dribbling",
            "football match general play",
            "players celebrating after goal",
            "soccer field wide view",
            "football players walking",
            "referee making decision",
            "crowd watching soccer match"
        ]

        # 编码文本描述（只需要做一次）
        self.register_buffer('shot_text_features', self._encode_texts(self.shot_descriptions))
        self.register_buffer('non_shot_text_features', self._encode_texts(self.non_shot_descriptions))

        # 冻结参数设置
        if freeze_vision:
            for param in self.clip_model.visual.parameters():
                param.requires_grad = False
        if freeze_text:
            for param in self.clip_model.transformer.parameters():
                param.requires_grad = False

        # 添加融合层和分类层
        self.fusion_layer = nn.Sequential(
            nn.Linear(512 * 3, 512),  # 图像特征 + 两类文本相似度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def _encode_texts(self, texts):
        """编码文本描述"""
        text_tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def forward(self, images):
        batch_size = images.shape[0]

        # 编码图像
        image_features = self.clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        # 计算与射门描述的相似度
        shot_similarities = torch.mm(image_features, self.shot_text_features.T)
        shot_max_sim = torch.max(shot_similarities, dim=1)[0]  # 取最大相似度
        shot_avg_sim = torch.mean(shot_similarities, dim=1)  # 取平均相似度

        # 计算与非射门描述的相似度
        non_shot_similarities = torch.mm(image_features, self.non_shot_text_features.T)
        non_shot_max_sim = torch.max(non_shot_similarities, dim=1)[0]
        non_shot_avg_sim = torch.mean(non_shot_similarities, dim=1)

        # 构造特征向量：[图像特征, 射门相似度差值, 非射门相似度差值]
        similarity_diff = shot_max_sim - non_shot_max_sim
        avg_similarity_diff = shot_avg_sim - non_shot_avg_sim

        # 融合特征
        fused_features = torch.cat([
            image_features,
            similarity_diff.unsqueeze(1),
            avg_similarity_diff.unsqueeze(1)
        ], dim=1)

        # 分类
        output = self.fusion_layer(fused_features)
        return output

    def get_detailed_predictions(self, images):
        """获取详细的预测信息，用于分析"""
        self.eval()
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            # 计算所有文本的相似度
            shot_similarities = torch.mm(image_features, self.shot_text_features.T)
            non_shot_similarities = torch.mm(image_features, self.non_shot_text_features.T)

            # 找出最相似的描述
            shot_best_idx = torch.argmax(shot_similarities, dim=1)
            non_shot_best_idx = torch.argmax(non_shot_similarities, dim=1)

            results = []
            for i in range(images.shape[0]):
                results.append({
                    'shot_best_match': self.shot_descriptions[shot_best_idx[i]],
                    'shot_similarity': shot_similarities[i, shot_best_idx[i]].item(),
                    'non_shot_best_match': self.non_shot_descriptions[non_shot_best_idx[i]],
                    'non_shot_similarity': non_shot_similarities[i, non_shot_best_idx[i]].item()
                })

            return results


class ZeroShotCLIPShotDetector(nn.Module):
    """零样本版本，不需要训练直接使用"""

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        # 统一转 fp32，避免 CUDA fp16 带来的类型/数值问题
        self.clip_model = self.clip_model.float()

        # 更精确的文本提示
        self.shot_prompts = [
            "a photo of a soccer shot",
            "a photo of a football player shooting at goal",
            "a photo of a goal attempt in soccer",
            "a photo of a soccer ball flying towards goal"
        ]

        self.non_shot_prompts = [
            "a photo of soccer players passing",
            "a photo of football players running",
            "a photo of general soccer gameplay",
            "a photo of soccer field action without shooting"
        ]

        # 预编码文本
        self.register_buffer('shot_text_features', self._encode_texts(self.shot_prompts))
        self.register_buffer('non_shot_text_features', self._encode_texts(self.non_shot_prompts))

    def _encode_texts(self, texts):
        text_tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def forward(self, images):
        with torch.no_grad():
            # 编码图像
            image_features = self.clip_model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            # 计算相似度
            shot_similarities = torch.mm(image_features, self.shot_text_features.T)
            non_shot_similarities = torch.mm(image_features, self.non_shot_text_features.T)

            # 取平均相似度作为分数
            shot_scores = torch.mean(shot_similarities, dim=1)
            non_shot_scores = torch.mean(non_shot_similarities, dim=1)

            # 返回射门的概率
            probabilities = torch.softmax(torch.stack([non_shot_scores, shot_scores], dim=1), dim=1)
            return probabilities[:, 1].unsqueeze(1)  # 返回射门类别的概率