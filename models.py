import torch
from torch import nn
from transformers import DebertaV2Model, DebertaV2PreTrainedModel


class DebertaV3ForCustomClassification(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)  # 使用 DebertaV2 作为基础模型
        self.dropout = nn.Dropout(0.1)  # 添加一个 dropout 层
        self.classifier = nn.Linear(
            config.hidden_size, config.num_labels)  # fully connected 层
        self.config = config  # 保存配置

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # 获取 DeBERTaV3 的输出
        outputs = self.deberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # 使用 Mean Pooling
        # [batch_size, seq_len, hidden_size]
        last_hidden_state = outputs.last_hidden_state
        # [batch_size, hidden_size]
        pooled_output = torch.mean(last_hidden_state, dim=1)

        # Dropout and Fully Connected Layer
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]

        # 如果提供了标签，则计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else logits
