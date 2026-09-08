import torch
import torch.nn as nn
import torch.nn.functional as F

class HAN_Classification(nn.Module):
    def __init__(self, 
                 embedding_dim=768,
                 gru_hidden_dim=128,
                 gru_num_layers=1,
                 prediction_hidden_dim=64,
                 num_classes=3,  # 修改点1：增加分类数参数
                 dropout=0.3):
        super(HAN_Classification, self).__init__()

        # ========================================
        # Layer 1: News-level Attention (文本级注意力)
        # 学习同一天内不同新闻的权重
        # ========================================
        self.news_attn_linear = nn.Linear(embedding_dim, 128)
        # 这个 context_vector (u_w) 是 HAN 的灵魂，它是模型“学习”出来的代表性新闻特征
        self.news_context_vector = nn.Parameter(torch.randn(128, 1)) 
        
        # ========================================
        # Layer 2: Temporal (Day-level) Bi-GRU
        # ========================================
        self.bi_gru = nn.GRU(
            input_size=embedding_dim,      
            hidden_size=gru_hidden_dim,    
            num_layers=gru_num_layers,    
            batch_first=True,
            bidirectional=True, 
            dropout=dropout if gru_num_layers > 1 else 0
        )

        # ========================================
        # Layer 3: Temporal Attention (日期级注意力)
        # ========================================
        self.temporal_attn_linear = nn.Linear(gru_hidden_dim * 2, 128)
        self.temporal_context_vector = nn.Parameter(torch.randn(128, 1))

        # ========================================
        # Layer 4: Prediction (Classification Head)
        # ========================================
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, prediction_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prediction_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 修改点2：最后一层改为 num_classes
            nn.Linear(32, num_classes) 
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        # 初始化 Context Vector 对收敛很重要
        nn.init.xavier_uniform_(self.news_context_vector)
        nn.init.xavier_uniform_(self.temporal_context_vector)

    def news_level_attention(self, x, mask):
        """
        x: (BW, L, E) -> (Batch*Window, News_Limit, Embedding)
        mask: (BW, L)
        """
        # 1. 线性变换得到隐藏表示 u_it
        u = torch.tanh(self.news_attn_linear(x)) # (BW, L, 128)
        
        # 2. 计算隐藏表示与“新闻级上下文向量”的相似度 (即权重得分)
        # 这一步是在学习：相对于整体而言，这篇新闻有多重要
        att_score = torch.matmul(u, self.news_context_vector).squeeze(-1) # (BW, L)
        
        if mask is not None:
            # Mask 掉没有新闻的填充位
            att_score = att_score.masked_fill(~mask, -1e9)
            is_all_masked = (~mask).all(dim=1, keepdim=True)
            att_score = att_score.masked_fill(is_all_masked, 0.0)
        
        # 3. Softmax 归一化
        att_weights = F.softmax(att_score, dim=1) # (BW, L)
        
        if mask is not None:
            att_weights = att_weights.masked_fill(is_all_masked, 0.0)
        
        # 4. 加权求和得到“日向量”
        day_vector = torch.sum(x * att_weights.unsqueeze(-1), dim=1) # (BW, E)
        return day_vector, att_weights

    def forward(self, x_text, x_mask):
        # x_text: (Batch, Window, L, E)
        B, W, L, E = x_text.size()
        
        # 1. 并行处理所有日期
        x_flat = x_text.view(B * W, L, E)
        mask_flat = x_mask.view(B * W, L)
        
        # 2. 文本级注意力聚合成日向量
        day_vectors_flat, news_weights = self.news_level_attention(x_flat, mask_flat)
        day_vectors_flat = self.dropout(day_vectors_flat)
        
        # 3. 恢复序列结构
        day_vectors = day_vectors_flat.view(B, W, E) # (B, W, E)
        
        # 4. 时间序列建模 (Bi-GRU)
        gru_out, _ = self.bi_gru(day_vectors) # (B, W, Hidden*2)
        
        # 5. 日期级注意力 (Temporal Attention)
        v_t = torch.tanh(self.temporal_attn_linear(gru_out)) # (B, W, 128)
        att_score_t = torch.matmul(v_t, self.temporal_context_vector).squeeze(-1) # (B, W)
        att_weights_t = F.softmax(att_score_t, dim=1) # (B, W)
        
        # 6. 加权求和得到最终特征向量
        doc_vector = torch.sum(gru_out * att_weights_t.unsqueeze(-1), dim=1) # (B, Hidden*2)
        
        # 7. 分类输出
        logits = self.fc(doc_vector) # (B, num_classes)
       
        # 注意：分类任务通常不在这里加 Softmax，因为 nn.CrossEntropyLoss 会自动处理
        return logits, att_weights_t, news_weights.view(B, W, L)
