"""XLM-R document detector; deterministic full-document window aggregation."""
import math
from pathlib import Path
import time
import numpy as np
import torch
from scipy.special import expit
from transformers import AutoModelForSequenceClassification,AutoTokenizer
import unicodedata,math
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).split())
def windows(ids):
 n=max(1,math.ceil(len(ids)/382))
 return [[0]+ids[len(ids)*i//n:len(ids)*(i+1)//n]+[2] for i in range(n)]


class XLMRDetector:
    def __init__(self,directory,device='cpu',threads=4):
        torch.set_num_threads(threads)
        self.device=torch.device(device)
        self.tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(
            directory,local_files_only=True,attn_implementation='sdpa').to(self.device).eval()
        self.thresholds=None
        path=Path(directory)/'thresholds.json'
        if path.exists():
            import json
            self.thresholds=json.loads(path.read_text(encoding='utf-8'))

    @torch.inference_mode()
    def score_ids(self,documents,batch_size=16):
        slices=[];owners=[]
        for i,ids in enumerate(documents):
            w=windows(ids);slices.extend(w);owners.extend([i]*len(w))
        logits=np.zeros(len(documents),dtype=float);counts=np.zeros(len(documents),dtype=int)
        for begin in range(0,len(slices),batch_size):
            batch=self.tokenizer.pad({'input_ids':slices[begin:begin+batch_size]},
                                      padding=True,pad_to_multiple_of=8,return_tensors='pt')
            batch={k:v.to(self.device) for k,v in batch.items()}
            values=self.model(**batch).logits.float()
            values=(values[:,1]-values[:,0]).cpu().numpy()
            np.add.at(logits,owners[begin:begin+len(values)],values)
            np.add.at(counts,owners[begin:begin+len(values)],1)
        return logits/counts,counts

    def score_texts(self,texts,batch_size=16):
        ids=self.tokenizer([norm(t) for t in texts],add_special_tokens=False,truncation=False)['input_ids']
        return self.score_ids(ids,batch_size)

    def check(self,text,language,batch_size=16):
        started=time.perf_counter()
        if self.thresholds is None:raise RuntimeError('Thresholds have not been calibrated')
        if language not in self.thresholds['languages']:raise ValueError('Unsupported language')
        text=norm(text)
        if sum(c.isalpha() for c in text)<12:
            return dict(language=language,score=None,decision='abstain',reason='too_few_letters')
        score,count=self.score_texts([text],batch_size)
        threshold=self.thresholds['languages'][language]['logit_threshold']
        return dict(language=language,score=float(expit(score[0])),logit=float(score[0]),
                    threshold=float(expit(threshold)),logit_threshold=threshold,windows=int(count[0]),
                    decision='ai_candidate' if score[0]>=threshold else 'below_threshold',
                    ms=(time.perf_counter()-started)*1000)
