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


from sklearn.metrics import mean_squared_error, r2_score


# In[3]:


full_text_train = utils.load_citation_abstracts_split('train')
full_text_val = utils.load_citation_abstracts_split('val')
full_text_test = utils.load_citation_abstracts_split('test')


# In[4]:


full_text_train += full_text_val


# In[5]:


tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')


# In[15]:


x_train = tokenizer([x['text'] for x in full_text_train], padding=True, truncation=True, max_length=2048)
y_train = [float(x['label']) for x in full_text_train]


# In[16]:


torch.cuda.memory_allocated()


# In[17]:


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


# In[18]:


train_dataset = FullTextDataset(x_train, y_train)


# In[10]:


torch.cuda.memory_allocated()


# In[20]:


EPOCHS = 8
BATCH_SIZE = 4
LR = 1e-5
DEVICE = torch.device('cuda')
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', gradient_checkpointing=True, num_labels=1).to(DEVICE)


# In[21]:


torch.cuda.memory_allocated()


# In[22]:


model.train()

total_steps = (len(train_dataset) // BATCH_SIZE) + 1
log_step = total_steps // 10
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

optim = AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    print(f'epoch: {epoch} / {EPOCHS}')
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


# In[23]:


model.save_pretrained('citation_model')
print(f'saved to citation_model/')


# In[24]:


x_test = tokenizer([x['text'] for x in full_text_test], padding=True, truncation=True, max_length=2048)
y_test = [float(x['label']) for x in full_text_test]
test_pids = [x['paper_id'] for x in full_text_test]


# In[25]:


test_dataset = FullTextDataset(x_test, y_test)


# In[26]:


# model.cuda()
model.eval()

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

preds = []
test_labels = []
for ix, batch in enumerate(test_loader):
    input_ids = batch['input_ids'].to(DEVICE)
    attention_mask = batch['attention_mask'].cuda()
    outputs = model(input_ids, attention_mask=attention_mask)
    preds.extend(outputs.logits.cpu().detach().numpy())
    test_labels.extend(batch['labels'].cpu().detach().numpy())
print(f'r2_score on test: {r2_score(test_labels, preds)}')


# In[28]:


filename = f'citation_model_outputs/{datetime.now().isoformat().replace(".", "_").replace(":", "-")}.json'
preds = list(map(float, preds))
test_labels = list(map(float, test_labels))
json.dump({'preds': preds, 'labels': test_labels, 'y_test': y_test, 'paper_ids': test_pids}, open(filename, 'w+'))


# In[ ]:




