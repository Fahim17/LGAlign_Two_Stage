from datetime import datetime
import math
import pandas as pd
import torch
import csv
import os
from haversine import haversine




def save_exp(emb_dim, loss_id, ln_rate, batch, epc, ls_mrgn, lbl_sm, dt_nm, trn_sz, val_sz, mdl_nm, msg, adp_nm):

    filepath = f'logs/log_{loss_id}.txt'
    with open(filepath, 'w') as file:
        file.write(f'\nHyperparameter info: {datetime.now()}' + "\n\n")
        file.write(f'Message: {msg}\n')
        file.write(f'Exp ID: {loss_id}\n')
        file.write(f'Embedded dimension: {emb_dim}\n')
        file.write(f'Learning rate: {ln_rate}\n')
        file.write(f'Batch Size: {batch}\n')
        file.write(f'Loss Margin: {ls_mrgn}\n')
        file.write(f'Label_Smoothing: {lbl_sm}\n')
        file.write(f'Epoch: {epc}\n')
        file.write(f'Dataset: {dt_nm}\n')
        file.write(f'Training Size: {trn_sz}\n')
        file.write(f'Validation Size: {val_sz}\n')
        file.write(f'Model Name: {mdl_nm}\n')
        file.write(f'Adapter_id: {adp_nm}\n')
        file.write('\n\n')
    
        # df.to_string(file, index=True)


def write_to_file(expID, msg, content):

    filepath = f'logs/log_{expID}.txt'
    with open(filepath, 'a') as file:
        file.write(f'\n{msg}')
        file.write(f'{content}\n')
    



def write_to_rank_file(expID, step, row):
    # Check if the file exists
    file_path = f'rank/rank_{expID}.csv'
    file_exists = os.path.isfile(file_path)

    row = row.tolist()
    row.insert(0, step)
    
    # Open the file in append mode ('a'), if the file doesn't exist, it will be created
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # If the file doesn't exist, you might want to write the header
        if not file_exists:
            # Assuming the first row of the data to be added is the header
            header = ["epoch", "top1", "top5", "top10", "top1%"]  # Modify this according to your header
            writer.writerow(header)
        
        # Write the row to the CSV file
        writer.writerow(row)




# Example usage:
# Assuming you have a DataFrame named 'df' and you want to save it to 'data.txt'
# save_dataframe_to_txt(df, 'data.txt')


def hyparam_info(emb_dim, loss_id, ln_rate, batch, epc, ls_mrgn, trn_sz, val_sz, mdl_nm):
    print('\nHyperparameter info:')
    print(f'Exp ID: {loss_id}')
    print(f'Embedded dimension: {emb_dim}')
    print(f'Learning rate: {ln_rate}')
    print(f'Batch Size: {batch}')
    print(f'Loss Margin: {ls_mrgn}')
    print(f'Epoch: {epc}')
    print(f'Training Size: {trn_sz}')
    print(f'Validation Size: {val_sz}')
    print(f'Model Name: {mdl_nm}')
    print('\n')

def get_rand_id():
    dt = datetime.now()
    return f"{math.floor(dt.timestamp())}"[2:]



def save_tensor(var_name,  var):
    torch.save(var, f'logs/save_in/{var_name}.pt')

def idsToDist(id_a, id_b, ll_csv):
    # latlong_a = (ll_csv['lat'][id_a], ll_csv['long'][id_a])
    # latlong_b = (ll_csv['lat'][id_b], ll_csv['long'][id_b])
    # dist_ab = haversine(latlong_a, latlong_b)
    data_path = '/data/Research/Dataset/CVUSA_Cropped/CVUSA'
    test_data = pd.read_csv(f'{data_path}/splits/val-19zl.csv', header=None)
    tv_all_data = pd.read_csv(f"{data_path}/split_locations/tv_all.csv", header=None)

    sim_id_a = int(test_data[0][id_a].split("/")[-1][:-4])
    tv_sim_id_b = int(tv_all_data[0][id_b].split("/")[-1][:-4])


    lat1 = ll_csv['lat'][sim_id_a]
    lon1 = ll_csv['long'][sim_id_a]

    lat2 = ll_csv['lat'][tv_sim_id_b]
    lon2 = ll_csv['long'][tv_sim_id_b]


    R = 6371  
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    dist_ab = R * c
    # print(dist_ab)
    # dist_ab = haversine((lat1, lon1), (lat2, lon2))


    return dist_ab 

# def save_weights(mdl, pth = 'model_weights/'):
#     print("Model's state_dict:")
#     for param_tensor in mdl.state_dict():
#         print(param_tensor, "\t", mdl.state_dict()[param_tensor].size())

#     # Save the model's state_dict to a text file
#     state_dict = mdl.state_dict()

#     # Convert the state_dict to a human-readable format
#     formatted_state_dict = {k: v.tolist() for k, v in state_dict.items()}

#     # Write the formatted state_dict to a text file
#     with open(f"{pth}model_weights.txt", "w") as f:
#         for key, value in formatted_state_dict.items():
#             f.write(f"{key}: {value}\n")

#     print("Model weights have been saved to model_weights.txt")



# make negative only from anchor or positive
def create_neg_keys(P):
    B, D = P.shape
    N = torch.zeros((B, B-1, D), dtype=P.dtype, device=P.device)  # Initialize the tensor

    for i in range(B):
        N[i] = torch.cat((P[:i], P[i+1:]), dim=0)  # Exclude P[i] and concatenate the rest

    return N

# make negative both from anchor and positive
def create_neg_keys_2(A, P):
    B, D = P.shape

    M = torch.zeros((B, B-1, D), dtype=A.dtype, device=A.device)  # Initialize the tensor

    for i in range(B):
        M[i] = torch.cat((A[:i], A[i+1:]), dim=0)  # Exclude A[i] and concatenate the rest


    N = torch.zeros((B, B-1, D), dtype=P.dtype, device=P.device)  # Initialize the tensor

    for j in range(B):
        N[j] = torch.cat((P[:j], P[j+1:]), dim=0)  # Exclude P[i] and concatenate the rest

        

    return torch.cat((M, N), dim=1)


# make negative from anchor and positive, and negative feature
def create_neg_keys_3(A, P, NN):
    B, D = P.shape

    M = torch.zeros((B, B-1, D), dtype=A.dtype, device=A.device)  # Initialize the tensor

    for i in range(B):
        M[i] = torch.cat((A[:i], A[i+1:]), dim=0)  # Exclude A[i] and concatenate the rest


    N = torch.zeros((B, B-1, D), dtype=P.dtype, device=P.device)  # Initialize the tensor

    for j in range(B):
        N[j] = torch.cat((P[:j], P[j+1:]), dim=0)  # Exclude P[i] and concatenate the rest


    O = torch.zeros((B, B-1, D), dtype=NN.dtype, device=NN.device)  # Initialize the tensor

    for k in range(B):
        O[k] = torch.cat((NN[:k], NN[k+1:]), dim=0)  # Exclude P[i] and concatenate the rest

        

    return torch.cat((M, N, O), dim=1)