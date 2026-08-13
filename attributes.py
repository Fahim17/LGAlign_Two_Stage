import torch
from helper_func import get_rand_id

# openai/clip-vit-base-patch32
# openai/clip-vit-large-patch14


class Configuration:
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_set_device = 0

    # Model
    model_name: str = '--'
    v_pretrain_weight: str = 'openai/clip-vit-large-patch14'
    t_pretrain_weight: str = 'openai/clip-vit-large-patch14'

    expID = -1
    embed_dim: int = 768 #CLS:512 or 768, patch:1024,
    save_weights = False

    # Adapters
    v_adapter_id = "ybelkada/opt-350m-lora"
    t_adapter_id = "ybelkada/opt-350m-lora"
    v_use_adapter = True
    t_use_adapter = True
    use_ptrain_adapter = True

    # Training
    epochs: int = 20
    lr = 0.00001
    batch_size: int = 20
    lang_with: str = 'sat' # 'sat' or 'None'
    train_eval_per_epoch = 2
    use_mixed_precision = True

    # Eval
    save_vis_embed = False
    use_vis_embed = True
    gnd_embed = []
    sat_embed = []
    gnd_embed_pretrn = torch.load("embeddings/clip_cvusa_gnd.pt")
    sat_embed_pretrn = torch.load("embeddings/clip_cvusa_sat.pt")
    batch_no = 0
    eval_size = -1

    # Data
    dataset_nm = "CVUSA" #CVUSA or #CVACT or #VIGOR or #GAMa
    eval_db = "CVUSA"
    lang = 'T1' # T1, T2 or T3
    use_neg_text = False
    num_workers = 5

    # Loss
    loss_margin = 1 # TripletMarginLoss
    label_smoothing=0.5 # Contrastive Loss

    #others
    msg: str = f'{lang} gpt-4o; ConCat, lang_with: {lang_with}, Test DB: {eval_db}, Exhibit 3.1;'
    latlong_csv = ''
