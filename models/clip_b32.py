import torch
import torch.nn as nn
import torchvision.models as models
import clip
# from transformers import CLIPModel
from transformers import CLIPVisionModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPModel, ResNetModel 
from attributes import Configuration as hypm
import torch.nn.functional as F
import timm
from peft import LoraConfig, get_peft_model


# # OG CLIP
# def getClipVisionModelRN():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("RN50", device=hypm.device)
#     for param in model.parameters():
#         param.requires_grad = False

#     return model.encode_image

# def getClipTextModelRN():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("RN50", device=hypm.device)
#     for param in model.parameters():
#         param.requires_grad = False

#     return model.encode_text


# OG CLIP with adapter
def getClipVisionModelRN():
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # base_model = models.resnet18(pretrained=True)
    # base_model.fc = nn.Linear(base_model.fc.in_features, 512)
    # # Define PEFT configuration (LoRA example)
    # peft_config = LoraConfig(
    #     r=8,                    # Rank of LoRA matrix
    #     lora_alpha=16,          # Alpha scaling factor
    #     lora_dropout=0.1,       # Dropout for LoRA layers
    #     target_modules=["fc"],  # Apply LoRA to the fully connected layer
    #     bias="none"
    # )

    # model = get_peft_model(base_model, peft_config).to(hypm.device)

    # for param in model.parameters():
    #     param.requires_grad = False
# ----------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _ = clip.load("RN50", device=device, jit=False)

    # Extract the ResNet-50 visual backbone
    base_model = clip_model.visual

    # Remove the final projection layer to get raw features (1024-d instead of 512)
    class CLIPFeatureExtractor(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            x = self.base_model(x)  # Shape: (N, 1024)
            return x

    feature_extractor = CLIPFeatureExtractor(base_model)

    # Define PEFT configuration (applying LoRA to deeper layers)
    peft_config = LoraConfig(
        r=8,                  
        lora_alpha=16,        
        lora_dropout=0.1,     
        target_modules=["conv1"],  # Apply LoRA to ResNet blocks
        bias="none"
    )

    # Wrap the model with PEFT
    model = get_peft_model(feature_extractor, peft_config)

    return model

def getClipTextModelRN():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=hypm.device)
    for param in model.parameters():
        param.requires_grad = False

    return model.encode_text
# --------------------------------------------------------------------------------------------------------------------------
# # HuggingFace CLIP
# def getClipVisionModel():
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
#     # model_v = CLIPModel.from_pretrained(hypm.v_pretrain_weight).to(hypm.device)
#     model_v = CLIPVisionModelWithProjection.from_pretrained(hypm.v_pretrain_weight).to(hypm.device)
#     # print(f'Model Device:{model_v.device}')
#     for param in model_v.parameters():
#         param.requires_grad = False

#     return model_v


# # HuggingFace CLIP
# def getClipTextModel():
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     model_t = CLIPTextModelWithProjection.from_pretrained(hypm.t_pretrain_weight).to(hypm.device)
#     # print(f'Model Device:{model_t.device}')
#     for param in model_t.parameters():
#         param.requires_grad = False

#     return model_t
# --------------------------------------------------------------------------------------------------------------------------

# HuggingFace CLIP
def getClipVisionModel():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if(hypm.v_use_adapter):
        model_v = CLIPVisionModelWithProjection.from_pretrained(hypm.v_pretrain_weight).to(hypm.device)
        if(hypm.use_ptrain_adapter):
            model_v.load_adapter(hypm.v_adapter_id, is_trainable=True)
        else:
            config_lora = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["classifier"],
            )
            model_v.add_adapter(adapter_name="lora_adapter", adapter_config=config_lora)

    else:
        model_v = CLIPVisionModelWithProjection.from_pretrained(hypm.v_pretrain_weight).to(hypm.device)
        # print(f'Model Device:{model_v.device}')
        for param in model_v.parameters():
            param.requires_grad = False

    return model_v


# HuggingFace CLIP
def getClipTextModel():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if(hypm.v_use_adapter):
        model_t = CLIPTextModelWithProjection.from_pretrained(hypm.t_pretrain_weight).to(hypm.device)
        if(hypm.use_ptrain_adapter):
            model_t.load_adapter(hypm.t_adapter_id, is_trainable=True)
        else:
            config_lora = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["classifier"],
            )
            model_t.add_adapter(adapter_name="lora_adapter", adapter_config=config_lora)

    else:
        model_t = CLIPTextModelWithProjection.from_pretrained(hypm.t_pretrain_weight).to(hypm.device)
        # print(f'Model Device:{model_t.device}')
        for param in model_t.parameters():
            param.requires_grad = False

    return model_t
# --------------------------------------------------------------------------------------------------------------------------

def getTransformerEncoder(dim_in, nhead = 3, dim_feedforward = 2048, dropout = 0.1, num_layers = 1):

    encoder_layer = torch.nn.TransformerEncoderLayer(dim_in, nhead, dim_feedforward, dropout)
    
    transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers)

    return transformer_encoder


def getCrossAttention(dim_in, d_out_kq = 768, d_out_v = 512):

    # crossattn = CrossAttention(dim_in, d_out_kq, d_out_v)
    crossattn = CrossAttention(dim_in, 1)


    return crossattn

# HuggingFace Eva-CLIP
def getClipVisionModelEVA():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_v = timm.create_model('eva_giant_patch14_clip_224', pretrained=True,).to(hypm.device)
    # print(f'Model Device:{model_v.device}')
    for param in model_v.parameters():
        param.requires_grad = False

    return model_v


# def __init__(self, emb_dim = 512):
#     super(ResNet, self).__init__()
#     self.modelName = 'ResNet18'
#     self.q_net = resnet18(ResNet18_Weights.IMAGENET1K_V1)
#     self.ref_net = resnet18(ResNet18_Weights.IMAGENET1K_V1)
#     # for param in self.q_net.parameters():
#     #     param.requires_grad = False
#     # for param in self.ref_net.parameters():
#     #     param.requires_grad = False
#     self.resnet_output = self.q_net.fc.out_features
#     # self.fc_q = nn.Linear(self.resnet_output, emb_dim)
#     # self.fc_r = nn.Linear(self.resnet_output, emb_dim)
#     # self.sigmoid = nn.Sigmoid()




# class CrossAttention(torch.nn.Module):
#     def __init__(self, d_in, d_out_kq, d_out_v):
#         super().__init__()
#         self.d_out_kq=d_out_kq
#         self.W_query= torch.nn.Parameter(torch.rand(d_in, d_out_kq))
#         self.W_key  = torch.nn.Parameter(torch.rand(d_in, d_out_kq))
#         self.W_value= torch.nn.Parameter(torch.rand(d_in, d_out_v))
    
#     def forward(self, x_1, x_2):
#         queries_1=x_1.matmul(self.W_query)
#         keys_2=x_2.matmul(self.W_key)
#         values_2=x_2.matmul(self.W_value)
        
#         attn_scores=queries_1.matmul(keys_2.T)
#         attn_weights=torch.softmax(
#             attn_scores/self.d_out_kq**0.5, dim=-1
#         )
        
#         context_vec=attn_weights.matmul(values_2)
#         return context_vec
    
class CrossAttention(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, query, key, value):
        # Apply multi-head attention
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        return attn_output, attn_weights