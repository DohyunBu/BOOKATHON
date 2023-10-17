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
model = AutoModelWithLMHead.from_pretrained("./fine_tuning/first_model/").to(device, dtype=torch.float16)

print("user input : ")
prompt=input()
prompt_ids = tokenizer.encode(prompt)
inp = tensor(prompt_ids)[None].cuda()
while True:
    preds = model.generate(inp,
                           max_length=1024,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           repetition_penalty=2.0,
                           do_sample = True, top_k=30, top_p=0.9,       
                           use_cache=True
                          )
    print(tokenizer.decode(preds[0].cpu().numpy()))
    print("save it?(yes=1 no=0)")

    s_temp = int(input())
    if s_temp==1:
        break

prompt = tokenizer.decode(preds[0].cpu().numpy())

f = open("./test_model/zero_model.txt","w")
f.write(prompt)
f.close()
