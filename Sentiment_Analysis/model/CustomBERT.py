import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForSequenceClassification, BertConfig

class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, with_gate = True):
        super(FANLayer, self).__init__()
        self.input_linear_p = nn.Linear(input_dim, output_dim//4, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, (output_dim-output_dim//2))
        self.activation = nn.GELU()        
        if with_gate:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))
    
    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        
        if not hasattr(self, 'gate'):
            output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        else:
            gate = torch.sigmoid(self.gate)
            output = torch.cat((gate*torch.cos(p), gate*torch.sin(p), (1-gate)*g), dim=-1)
        return output

class CustomBertClassifier(BertForSequenceClassification):
    def __init__(self, num_labels=2, num_hidden_layers=12, replace_ffn=False, with_gate=False):
        config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels)
        config.num_hidden_layers = num_hidden_layers
        super(CustomBertClassifier, self).__init__(config)
        if replace_ffn: # replace the two linear layers in FFN for each layer
            for layer in self.bert.encoder.layer:
                layer.intermediate = BertIntermediate_withFAN(config) # replace the intermediate layer because we don't need the activation function within the bert intermediate layer, which is already implemented in the FANLayer
                layer.output.dense = FANLayer(config.intermediate_size, config.hidden_size, with_gate=with_gate)
                
class BertIntermediate_withFAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = FANLayer(config.hidden_size, config.intermediate_size, p_ratio=config.p_ratio, with_gate=config.with_gate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return hidden_states
