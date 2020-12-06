#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import json
import utils
from datetime import datetime
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import LongformerTokenizer, LongformerForSequenceClassification, AdamW


# In[2]:


full_text_train = utils.load_top_class_abstracts_split('train')
full_text_val = utils.load_top_class_abstracts_split('val')
full_text_test = utils.load_top_class_abstracts_split('test')


# In[3]:


full_text_train += full_text_val


# In[4]:


tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')


# In[5]:


x_train = tokenizer([x['text'] for x in full_text_train], padding=True, truncation=True, max_length=2048)
y_train = [x['label'] for x in full_text_train]


# In[6]:


torch.cuda.memory_allocated()


# In[7]:


class FullTextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# In[8]:


train_dataset = FullTextDataset(x_train, y_train)


# In[9]:


torch.cuda.memory_allocated()


# In[10]:


EPOCHS = 6
BATCH_SIZE = 8
LR = 1e-5
DEVICE = torch.device('cuda')
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', gradient_checkpointing=True).to(DEVICE)
print(f'EPOCHS: {EPOCHS}, BATCH_SIZE: {BATCH_SIZE}, LR: {LR}')

# In[11]:


torch.cuda.memory_allocated()


# In[ ]:


model.train()

total_steps = (len(train_dataset) // BATCH_SIZE) + 1
log_step = total_steps // 10
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

optim = AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    print(f'epoch: {epoch}')
    for batch_ix, batch in tqdm(enumerate(train_loader), total=total_steps):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
        if batch_ix % log_step == 0:
            print(f'loss: {loss}')


# In[ ]:


filename = f'top_class_model/{datetime.now().isoformat().replace(".", "_").replace(":", "-")}'
model.save_pretrained(filename)
print(f'saved to {filename}')


# In[ ]:


x_test = tokenizer([x['text'] for x in full_text_test], padding=True, truncation=True, max_length=2048)
y_test = [x['label'] for x in full_text_test]
test_pids = [x['paper_id'] for x in full_text_test]


# In[ ]:


test_dataset = FullTextDataset(x_test, y_test)


# In[ ]:


# model.cuda()
model.eval()

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

preds = []
test_labels = []
for ix, batch in enumerate(test_loader):
    input_ids = batch['input_ids'].to(DEVICE)
    attention_mask = batch['attention_mask'].cuda()
    outputs = model(input_ids, attention_mask=attention_mask)
    preds.extend(outputs.logits.cpu().argmax(axis=1).detach().numpy())
    test_labels.extend(batch['labels'].cpu().detach().numpy())
print(f'f1 on test: {f1_score(test_labels, preds)}')


# In[ ]:


filename = f'top_class_model_outputs/{datetime.now().isoformat().replace(".", "_").replace(":", "-")}.json'
preds = list(map(int, preds))
test_labels = list(map(int, test_labels))
json.dump({'preds': preds, 'labels': test_labels, 'y_test': y_test, 'paper_ids': test_pids}, open(filename, 'w+'))

