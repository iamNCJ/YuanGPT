from torch import nn

from model import BaseModel

class ColAICriterion(nn.Module):
    def __init__(self, model: BaseModel):
        super().__init__()
        self.model = model
        self.loss_fct = nn.CrossEntropyLoss()

    def loss(self, logits, labels):
        
        # print(logits.shape, labels.shape)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # half = self.config.batch_size // 2
        # shift_logits0 = shift_logits[:half, ...]
        # shift_logits1 = shift_logits[half:, ...]
        # shift_labels0 = shift_labels[:half, ...]
        # shift_labels1 = shift_labels[half:, ...]
        # print(shift_logits0.shape, shift_labels0.shape)
        # res0 = self.loss_fct(shift_logits0.view(-1, shift_logits0.size(-1)), shift_labels0.view(-1))
        # res1 = self.loss_fct(shift_logits1.view(-1, shift_logits1.size(-1)), shift_labels1.view(-1))
        # print(shift_logits.shape, shift_labels.shape, shift_logits.device, shift_labels.device)
        res = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # print(res0.shape, res1.shape)
        # print(res0)quit()
        # assert torch.isnan(shift_logits).sum() == 0
        # assert torch.isnan(shift_labels).sum() == 0
        # assert torch.isnan(res).sum() == 0
        # return (res0 * half + res1 * (self.config.batch_size - half)) / self.config.batch_size
        return res

    def forward(self, logits, labels):
        return self.loss(logits, labels)
