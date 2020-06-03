import json
import os
import numpy as np
import pickle
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils

class Myrnn(nn.Module):
    def __init__(self, input_dim, hidden_size= 100):
        super(Myrnn, self).__init__()
        self.hidden_size = hidden_size

        self.Linear1 = nn.Linear(input_dim, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers= 5, bidirectional= True)
        self.Linear2 = nn.Linear(hidden_size* 2, 2)
        self.Linear3 = nn.Linear(hidden_size* 2, 1)

    def forward(self, input_data):
        out = F.relu(self.Linear1(input_data))
        out, hidden = self.rnn(out)
        #out1 is for onset & offset
        out1 = torch.sigmoid(self.Linear2(out))
        #out2 is for pitch
        out2 = self.Linear3(out)
        return out1, out2


class MyData(Data.Dataset):
    def __init__(self, data_seq, label):
        self.data_seq = data_seq
        self.label= label

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return {
            'data': self.data_seq[idx],
            'label': self.label[idx]
        }

def collate_fn(samples):
    batch = {}
    #print (samples[0]['data'].shape)
    temp= [torch.from_numpy(np.array(sample['data'], dtype= np.float32)) for sample in samples]
    padded_data = rnn_utils.pad_sequence(temp, batch_first=True, padding_value= 0)
    batch['data']= padded_data
    batch['label']= [np.array(sample['label'], dtype= np.float32) for sample in samples]

    return batch

def post_processing(output1, pitch):
    pitch= pitch.squeeze(1).squeeze(1).cpu().detach().numpy()
    print (pitch.shape)
    print (torch.mean(output1))
    threshold= 0.1
    notes= []
    this_onset= None
    this_offset= None
    this_pitch= None

    for i in range(len(output1)):
        if output1[i][0][0] > threshold and this_onset == None:
            this_onset= i
        elif output1[i][0][1] > threshold and this_onset != None and this_onset+ 1 < i and this_offset == None:
            this_offset= i
            this_pitch= int(round(np.mean(pitch[this_onset:this_offset+ 1])))
            notes.append([this_onset* 0.032+ 0.016, this_offset* 0.032+ 0.016, this_pitch])
            this_onset= None
            this_offset= None
            this_pitch= None

    print (np.array(notes))
    return notes

def testing(net, sample, device):
    net.eval()
    data = sample['data']
    data= torch.Tensor(data)

    target= sample['label']
    target= torch.Tensor(target)

    data= data.unsqueeze(1)
    target= target.unsqueeze(1)

    print (data.shape)
    print (target.shape)

    data_length= list(data.shape)[0]

    data = data.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)

    output1, output2 = net(data)
    print (output1.shape)
    print (output2.shape)
    answer= post_processing(output1, output2)
    return answer



def do_training(net, loader, optimizer, device):

    num_epoch = 50
    criterion_onset= nn.BCELoss()
    criterion_pitch= nn.L1Loss()
    train_loss= 0.0
    total_length= 0

    for epoch in range(num_epoch):
        net.train()
        total_length= 0.0
        print ("epoch %d start time: %f" %(epoch, time.time()))
        train_loss= 0.0

        for batch_idx, sample in enumerate(loader):
            data = sample['data']
            data= torch.Tensor(data)

            target= sample['label']
            target= torch.Tensor(target)


            data= data.permute(1,0,2)
            target= target.permute(1,0,2)

            #print (data.shape)
            #print (target.shape)
            data_length= list(data.shape)[0]

            data = data.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            optimizer.zero_grad()
            output1, output2 = net(data)
            #print (output1)
            #print (output2)

            #print (output1.shape)
            #print (output2.shape)

            total_loss= criterion_onset(output1, torch.narrow(target, dim= 2, start= 0, length= 2))
            total_loss= total_loss+ criterion_pitch(output2, torch.narrow(target, dim= 2, start= 2, length= 1))
            train_loss= train_loss+ total_loss.item()
            total_length= total_length+ 1
            total_loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print ("epoch %d, sample %d, loss %.6f" %(epoch, batch_idx, total_loss))
                #print ("current time: %f" %(time.time()))
                sys.stdin.flush()
            
            

        print('epoch %d, avg loss: %.6f' %(epoch, train_loss/ total_length))
        

        model_path= f'ST_{epoch}.pt'
        torch.save(net.state_dict(), model_path)  

    return net

def preprocess(data_seq, label):
    new_label= []
    for i in range(len(label)):
        label_of_one_song= []
        cur_note= 0
        cur_note_onset= label[i][cur_note][0]
        cur_note_offset= label[i][cur_note][1]
        cur_note_pitch= label[i][cur_note][2]

        for j in range(len(data_seq[i])):
            cur_time= j* 0.032+ 0.016
        
            if abs(cur_time - cur_note_onset) < 0.017:
                label_of_one_song.append(np.array([1, 0, cur_note_pitch]))

            elif cur_time < cur_note_onset or cur_note >= len(label[i]):
                label_of_one_song.append(np.array([0, 0, 0.0]))

            elif abs(cur_time - cur_note_offset) < 0.017:
                label_of_one_song.append(np.array([0, 1, cur_note_pitch]))
                cur_note= cur_note+ 1
                if cur_note < len(label[i]):
                    cur_note_onset= label[i][cur_note][0]
                    cur_note_offset= label[i][cur_note][1]
                    cur_note_pitch= label[i][cur_note][2]
            else:
                label_of_one_song.append(np.array([0, 0, cur_note_pitch]))

        new_label.append(label_of_one_song)

    return new_label


if __name__ == '__main__':

    THE_FOLDER = "./MIR-ST500"

    data_seq= []
    label= []
    
    for the_dir in os.listdir(THE_FOLDER):
        print (the_dir)
        if not os.path.isdir(THE_FOLDER + "/" + the_dir):
            continue

        json_path = THE_FOLDER + "/" + the_dir+ f"/{the_dir}_feature.json"
        gt_path= THE_FOLDER+ "/" +the_dir+ "/"+ the_dir+ "_groundtruth.txt"

        youtube_link_path= THE_FOLDER+ "/" + the_dir+ "/"+ the_dir+ "_link.txt"

        with open(json_path, 'r') as json_file:
            temp = json.loads(json_file.read())

        gtdata = np.loadtxt(gt_path)

        data= []
        for key, value in temp.items():
            data.append(value)

        data= np.array(data).T

        data_seq.append(data)
        label.append(gtdata)
    
    label= preprocess(data_seq, label)
    train_data = MyData(data_seq, label)
    
    print(label[0])
    print(train_data[0])
    
    
    # with open("feature_pickle.pkl", 'wb') as pkl_file:
    #     pickle.dump(train_data, pkl_file)
    
    # train_data= None
    # with open("feature_pickle.pkl", 'rb') as pkl_file:
    #     train_data= pickle.load(pkl_file)

    input_dim= 23
    hidden_size= 50

    BATCH_SIZE= 1
    loader = Data.DataLoader(dataset=train_data, batch_size= BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers = 12)

    model = Myrnn(input_dim, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr= 0.001)

    device = 'cpu'
    
    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'
    
    print("use",device,"now!")

    model.to(device)
    model= do_training(model, loader, optimizer, device)

    #for testing
    
    #model.load_state_dict(torch.load("ST_5.pt"))
    #testing(model, train_data[0], device)
