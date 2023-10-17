import tensorflow as tf
import torch
import transformers
from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast
from fastai.text.all import *
import fastai
import re
import pandas as pd
import grpc
from proto.message_log_pb2 import LogRequest
from proto.message_log_pb2 import NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
from proto.message_log_pb2_grpc import MessageLoggerStub

LOG_END_POINT = 'localhost:36000'

def request_log(request):
    with grpc.insecure_channel(LOG_END_POINT) as channel:
        stub = MessageLoggerStub(channel)
        response = stub.LogMessage(request)

        return response


tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>') 
device = torch.device("cuda")
model = AutoModelWithLMHead.from_pretrained("./4fine_tuning5/first_model/").to(device, dtype=torch.float16)

print("user input : ")
prompt=input()
prompt_ids = tokenizer.encode(prompt)
inp = tensor(prompt_ids)[None].cuda()

pos1=0
pos2=0
pos3=0
pos4=0
pos5=0
pos=0
i = 0
while(1):
    temp_length = len(prompt)
    pos = temp_length
    preds = model.generate(inp,
                           max_length=temp_length+100,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           repetition_penalty=2.0,
                           do_sample = True, top_k=30, top_p=0.8,
                           num_return_sequences=5,       
                           use_cache=True
                          )
#inp = tensor(tensor(tokenizer.encode(tokenizer.decode(preds[0].cpu().numpy())))[None].cuda())
    pos1 = pos+tokenizer.decode(preds[0].cpu().numpy())[pos+1:].find(". ")+2
    if pos1 == 1:
        pos1 = pos
    print("[1]: ",tokenizer.decode(preds[0].cpu().numpy())[:pos1])
    print()
    if pos2 == 1:
        pos2 = pos
    pos2 = pos+tokenizer.decode(preds[1].cpu().numpy())[pos+1:].find(". ")+2
    print("[2]: ",tokenizer.decode(preds[1].cpu().numpy())[:pos2])
    print()
    if pos3 == 1:
        pos3 = pos
    pos3 = pos+tokenizer.decode(preds[2].cpu().numpy())[pos+1:].find(". ")+2
    print("[3]: ",tokenizer.decode(preds[2].cpu().numpy())[:pos3])
    print()
    if pos4 == 1:
        pos4 = pos
    pos4 = pos+tokenizer.decode(preds[3].cpu().numpy())[pos+1:].find(". ")+2
    print("[4]: ",tokenizer.decode(preds[3].cpu().numpy())[:pos4])
    print()
    if pos5 == 1:
        pos5 = pos
    pos5 = pos+tokenizer.decode(preds[4].cpu().numpy())[pos+1:].find(". ")+2
    print("[5]: ",tokenizer.decode(preds[4].cpu().numpy())[:pos5])
    print()
    print("input?(retry = 0, end = 9, user_input = 7) ")
    temps = input()
    while temps<'0' and temps>'9':
        print("input error(retry)")
    temp1=int(temps)
    if temp1==7:
        print("[user_input] : ",prompt)
        prompt+=input()
        inp = tensor(tokenizer.encode(prompt))[None].cuda()
        continue
    elif temp1==0:
        continue
    elif temp1==9:
        print("end")
        break
    elif temp1>=1 and temp1<=5:
        if temp1==1:
            pos = pos1
        elif temp1==2:
            pos = pos2
        elif temp1==3:
            pos = pos3
        elif temp1==4:
            pos = pos4
        else:
            pos = pos5
        inp = tensor(tensor(tokenizer.encode(tokenizer.decode(preds[temp1-1].cpu().numpy())[:pos]))[None].cuda())
        msg = 'Prompt: {}\nGenearted: {}'.format(prompt, tokenizer.decode(preds[temp1-1].cpu().numpy())[:pos])
        log_request = LogRequest(log_level=DEBUG, msg=msg)

        log_response = request_log(log_request)
        assert log_response.done is True

        prompt = tokenizer.decode(preds[temp1-1].cpu().numpy())[:pos]
        f = open(f"./first_topic_model/jang/recording{i}.txt","w")
        f.write(prompt)
        f.close()
        i += 1
    else:
        print("input error(retry)")
        
print(prompt)

