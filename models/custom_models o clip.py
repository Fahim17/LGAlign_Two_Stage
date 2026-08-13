import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
# from vit_pytorch import ViT
from models.clip_b32 import getClipTextModel, getClipVisionModel, getClipVisionModelEVA, getTransformerEncoder, getCrossAttention, getClipTextModelRN, getClipVisionModelRN
from transformers import AutoTokenizer, AutoProcessor
import clip

from attributes import Configuration as hypm





# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, emb_dim = 512):
        super(ResNet, self).__init__()
        self.modelName = 'ResNet18'
        self.q_net = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        self.ref_net = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        # for param in self.q_net.parameters():
        #     param.requires_grad = False
        # for param in self.ref_net.parameters():
        #     param.requires_grad = False
        self.resnet_output = self.q_net.fc.out_features
        # self.fc_q = nn.Linear(self.resnet_output, emb_dim)
        # self.fc_r = nn.Linear(self.resnet_output, emb_dim)
        # self.sigmoid = nn.Sigmoid()





    def forward(self, q, r, isTrain = True, isQuery = True):
        xq = self.q_net(q)
        # xq = self.fc_q(xq)
        # xq = torch.sigmoid(xq)

        xr = self.ref_net(r)
        # xr = self.fc_r(xr)
        # xr = torch.sigmoid(xr)
        
        if isTrain:
            # print(f'dukse train')
            return xq, xr
            # return self.query.encode_image(q), self.ref.encode_image(r)
        else:
            if isQuery:
                # print(f'dukse query')
                return xq
                # return self.query.encode_image(q)
            else:
                # print(f'dukse ref')
                return xr
                # return self.ref.encode_image(r)

    



class ResNet2(nn.Module):
    def __init__(self, emb_dim):
        super(ResNet2, self).__init__()
        self.modelName = 'ResNet18'
        self.net = resnet18()



    def forward(self, img):
        return self.net(img)


# # Define the VIT model
# class VIT(nn.Module):
#     def __init__(self):
#         super(VIT, self).__init__()
#         self.modelName = 'VIT'
#         self.query = ViT(
#             image_size = 256,
#             patch_size = 32,
#             num_classes = 512,
#             dim = 1024,
#             depth = 6,
#             heads = 16,
#             mlp_dim = 2048,
#             dropout = 0.1,
#             emb_dropout = 0.1
#             )
#         self.ref = ViT(
#             image_size = 256,
#             patch_size = 32,
#             num_classes = 512,
#             dim = 1024,
#             depth = 6,
#             heads = 16,
#             mlp_dim = 2048,
#             dropout = 0.1,
#             emb_dropout = 0.1
#             )



#     def forward(self, q, r, isTrain = True, isQuery = True):
#         if isTrain:
#             return self.query(q), self.ref(r)
#         else:
#             if isQuery:
#                 return self.query(q)
#             else:
#                 return self.ref(r)




# Define the Hugging face CLIP model
class CLIP_model(nn.Module):
    def __init__(self, embed_dim):
        super(CLIP_model, self).__init__()
        self.modelName = 'CLIP'
        self.device = hypm.device
        self.tokenizer = AutoTokenizer.from_pretrained(hypm.t_pretrain_weight)
        self.processor = AutoProcessor.from_pretrained(hypm.v_pretrain_weight)


#-----------------------OG-ViT--------------------------
        self.query = getClipVisionModel()
        # for param in self.query.parameters():
        #     param.requires_grad = False

        self.ref = getClipVisionModel()
        # for param in self.ref.parameters():
        #     param.requires_grad = False

        self.text = getClipTextModel()
        # for param in self.text.parameters():
        #     param.requires_grad = False

#-----------------------Res50------------------------
        # self.query = getClipVisionModelRN()
        # self.ref = getClipVisionModelRN()
        # self.text = getClipTextModelRN()

#-----------------------OG-EVA--------------------------
        # self.query = getClipVisionModelEVA()
        # # for param in self.query.parameters():
        # #     param.requires_grad = False

        # self.ref = getClipVisionModelEVA()
        # # for param in self.ref.parameters():
        # #     param.requires_grad = False

        # self.text = getClipTextModel()
        # # for param in self.text.parameters():
        # #     param.requires_grad = False
#------------------------------------------------------



        # self.norm_shape = self.query.vision_model.post_layernorm.normalized_shape[0]
        

# -------------------------------------------og---------------------------------------------------------------------------       
        self.vis_embed_shape = self.query.visual_projection.out_features #og
        self.txt_embed_shape = self.text.text_projection.out_features

# -------------------------------------------Res50---------------------------------------------------------------------------
        # self.vis_embed_shape = 512
        # self.txt_embed_shape = 512

# -------------------------------------------og-EVA---------------------------------------------------------------------------       
        # self.vis_embed_shape = hypm.embed_dim
        # self.txt_embed_shape = self.text.text_projection.out_features
# ----------------------------------------------------------------------------------------------------------------------

        self.mlp_txt = nn.Linear(self.vis_embed_shape + self.txt_embed_shape, embed_dim).to(device=self.device) #concat

        # self.vis_sat_L1 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        # self.vis_gnd_L1 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        # self.txt_gnd_L1 = nn.Linear(embed_dim, embed_dim).to(device=self.device)



        self.vis_txt_L1 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        self.vis_txt_L2 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        self.vis_txt_L3 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        # self.vis_txt_L4 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        # self.vis_txt_L5 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        # self.vis_txt_L6 = nn.Linear(embed_dim, embed_dim).to(device=self.device)



        self.vis_L1 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        self.vis_L2 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        self.vis_L3 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        # self.vis_L4 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        # self.vis_L5 = nn.Linear(embed_dim, embed_dim).to(device=self.device)
        # self.vis_L6 = nn.Linear(embed_dim, embed_dim).to(device=self.device)

# ----------------------------------------------patch------------------------------------------------------------------------
        # self.patch_temp_q = nn.Linear(1792, embed_dim).to(device=self.device) #patch
        # self.patch_temp_r = nn.Linear(1792, embed_dim).to(device=self.device) #patch


# ----------------------------------------------Cross Attention------------------------------------------------------------------------
        
        # self.mlp_txt = getCrossAttention(dim_in=self.vis_embed_shape, d_out_v=self.vis_embed_shape).to(device=self.device) #cross-atten
# ----------------------------------------------encoder------------------------------------------------------------------------

        # self.trnfr_encoder_1 = torch.nn.TransformerEncoderLayer(embed_dim, 3, 2024, 0.1).to(device=self.device)
        # self.trnfr_encoder_2 = torch.nn.TransformerEncoderLayer(embed_dim, 3, 2024, 0.1).to(device=self.device)

        # self.trnfr_encoder_1 = getTransformerEncoder(dim_in = embed_dim).to(device=self.device)
        # self.trnfr_encoder_2 = getTransformerEncoder(dim_in = embed_dim).to(device=self.device)






    def get_vision_embeddings(self, imgs, isQ=True):
        # Preprocess the images
        temp_dic = {'pixel_values':imgs}
        # temp_dic = self.processor(images=imgs, return_tensors="pt")

        # Use the CLIP model to get vision embeddings
        
        # with torch.no_grad():
        # #------------------------og-------------------------------------
        if isQ:
            # hypm.gnd_embed.append(outputs.image_embeds.cpu())
            if(hypm.use_vis_embed):
                # startidx = hypm.batch_no*hypm.batch_size
                # endidx = ((hypm.batch_no*hypm.batch_size)+hypm.batch_size)
                # if(endidx>hypm.eval_size):
                #     image_embeds = hypm.gnd_embed_pretrn[startidx:,:].to(hypm.device)
                # else:
                #     image_embeds = hypm.gnd_embed_pretrn[startidx:endidx,:].to(hypm.device)

                # return image_embeds
                return imgs
            else:
                outputs = self.query(imgs)

        else:
            # hypm.sat_embed.append(outputs.image_embeds.cpu())
            if(hypm.use_vis_embed):
                # startidx = hypm.batch_no*hypm.batch_size
                # endidx = ((hypm.batch_no*hypm.batch_size)+hypm.batch_size)
                # if(endidx>hypm.eval_size):
                #     image_embeds = hypm.gnd_embed_pretrn[startidx:,:].to(hypm.device)
                # else:
                #     image_embeds = hypm.sat_embed_pretrn[startidx:endidx,:].to(hypm.device)
                # return image_embeds
                return imgs
            else:
                outputs = self.ref(imgs)



        # last_hidden_state = outputs.last_hidden_state
        # pooled_output = outputs.pooler_output  # pooled CLS states
        image_embeds = outputs.image_embeds

        #-------------------------patch------------------
        # if isQ:
        #     outputs = self.query(imgs)
        #     image_embeds = outputs.image_embeds

        #     outputs_patch = self.query.vision_model.embeddings(imgs)
        #     patch_only_embeddings = outputs_patch[:, 1:, :]
        #     average_patch_embedding = patch_only_embeddings.mean(dim=1)

        #     image_embeds = torch.cat((image_embeds, average_patch_embedding),1)
        #     image_embeds = self.patch_temp_q(image_embeds)
 
        # else:
        #     outputs = self.ref(imgs)
        #     image_embeds = outputs.image_embeds

        #     outputs_patch = self.ref.vision_model.embeddings(imgs)
        #     patch_only_embeddings = outputs_patch[:, 1:, :]
        #     average_patch_embedding = patch_only_embeddings.mean(dim=1)

        #     image_embeds = torch.cat((image_embeds, average_patch_embedding),1)
        #     image_embeds = self.patch_temp_r(image_embeds)

        #--------------------------Res50----------------------------

        # if isQ:
        #     outputs = self.query(imgs)
 
        # else:
        #     outputs = self.ref(imgs)
        # image_embeds = outputs
        # image_embeds = image_embeds.to(torch.float32) 

        # #------------------------og-EVA-------------------------------------
        # if isQ:
        #     outputs = self.query(imgs)
 
        # else:
        #     outputs = self.ref(imgs)
        # image_embeds = outputs
          
        #-----------------------------------------------------------
        
        
        
        return image_embeds
    
    def get_text_embeddings(self, txt):
        #--------------------------OG----------------------------
        txt = self.tokenizer(txt, padding=True, truncation=True, return_tensors="pt", max_length=77)
        txt.to(device=self.device)
        outputs = self.text(**txt)
        text_embeds = outputs.text_embeds
        #--------------------------Res50----------------------------
        # txt = clip.tokenize(txt, context_length=77, truncate=True)
        # txt = txt.to(device=self.device)
        # outputs = self.text(txt)
        # text_embeds = outputs
        # text_embeds = text_embeds.to(torch.float32)           
        #-----------------------------------------------------------

        return text_embeds


    def forward(self, q, r, t, isTrain = True, isQuery = True):
        xq = self.get_vision_embeddings(imgs = q, isQ = True )
        xr = self.get_vision_embeddings(imgs = r, isQ = False )

        hypm.batch_no+=1
        
        if(hypm.use_neg_text):
            xt = self.get_text_embeddings(txt = t[0])
            xt_n = self.get_text_embeddings(txt = t[1])
        else:
            xt = self.get_text_embeddings(txt = t)
            # shuffled_indices = torch.randperm(xt.size(0))
            # xt_n = xt[shuffled_indices]

        # xq = self.vis_gnd_L1(xq)
        # xr = self.vis_sat_L1(xr)
        # xt = self.txt_gnd_L1(xt)


        if (hypm.lang_with=='sat'):
            # --------------Concat---------------
            xlt = torch.cat((xr, xt), 1)
            xlt = self.mlp_txt(xlt)
            # --------------without text---------------
            # xlt = xr
            # --------------without ground---------------
            # xq = xt
            # xlt = xr
            #------------------------------------------
            # xlt_n = torch.cat((xr, xt_n), 1)
            # xlt_n = self.mlp_txt(xlt_n)




            # --------------Cross attention---------------
            # xr = xr.unsqueeze(1)
            # xt = xt.unsqueeze(1)
            # xlt, xlt_w = self.mlp_txt(xt, xr, xr)
            # xlt = xlt.squeeze(1)
            # # print(xlt.shape)
            # --------------Addition---------------
            # xlt = xr+xt
            # --------------Resnet50---------------
            # xlt = torch.cat((xr, xt), 1)
            # xlt = self.mlp_txt(xlt)
            # xq = xq.to(torch.float32)           
            

# -----------------------------------------------------------
            xlt = self.vis_txt_L1(xlt)
            xlt = self.vis_txt_L2(xlt)
            xlt = self.vis_txt_L3(xlt)
            # xlt = self.vis_txt_L4(xlt)
            # xlt = self.vis_txt_L5(xlt)
            # xlt = self.vis_txt_L6(xlt)



            xq = self.vis_L1(xq)
            xq = self.vis_L2(xq)
            xq = self.vis_L3(xq)
            # xq = self.vis_L4(xq)
            # xq = self.vis_L5(xq)
            # xq = self.vis_L6(xq)


            # xlt_n = self.vis_txt_L1(xlt_n)
            # xlt_n = self.vis_txt_L2(xlt_n)
            # xlt_n = self.vis_txt_L3(xlt_n)



# -----------------------------------------------------------

            # xlt = self.trnfr_encoder_1(xlt)
            # xq = self.trnfr_encoder_2(xq)


            if(hypm.use_neg_text):
                return xq, xlt, -1
            else:
                return xq, xlt, -1
        else:
            xr = self.vis_txt_L1(xr)
            xr = self.vis_txt_L2(xr)
            xr = self.vis_txt_L3(xr)

            xq = self.vis_L1(xq)
            xq = self.vis_L2(xq)
            xq = self.vis_L3(xq)

            return xq, xr


        # xr = self.ref_fc1(xr)
        # xr = torch.relu(xr)
        # xr = self.ref_fc2(xr)
        # xr = torch.sigmoid(xr)
        
        # if isTrain:
        #     # return xq, xr
        #     return xq, xlt
            # return self.query.encode_image(q), self.ref.encode_image(r)
        # else:
        #     return xq, xlt
            # if isQuery:
            #     return xq
            #     # return self.query.encode_image(q)
            # else:
            #     return xrt
            #     # return self.ref.encode_image(r)








# Define the ResNet model
# class VIT(nn.Module):
#     def __init__(self):
#         super(VIT, self).__init__()
#         self.modelName = 'VIT_B_16'
#         self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
#         # for param in self.vit.parameters():
#         #     param.requires_grad = False
#         # num_features = self.vit.heads
#         # self.vit.fc = nn.Linear(num_features, emb_dim)



#     def forward(self, x):
#         return self.vit(x)


# # Define the ResNet model
# class ResNet(nn.Module):
#     def __init__(self, emb_dim):
#         super(ResNet, self).__init__()
#         self.resnet = torchvision.models.resnet50(pretrained=True)
#         num_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_features, emb_dim)

#     def forward(self, x):
#         return self.resnet(x)







