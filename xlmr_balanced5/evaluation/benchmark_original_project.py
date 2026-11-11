from pathlib import Path
import sys,json,time,hashlib,platform
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,precision_recall_curve,average_precision_score,auc
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/xlmr_delivery_20261111';B=OUT/'xlmr_balanced5';sys.path.insert(0,str(B))
from detector import XLMRDetector
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
data=Path('D:/CodexData/xlmr_reviews5_20261111/test.parquet');test=pd.read_parquet(data)
ref=pd.read_parquet(ROOT/'outputs/xlmr_reviews5_20261111/predictions.parquet').set_index('id')
assert len(test)==4213 and test.id.is_unique
config=json.loads((B/'model/thresholds.json').read_text(encoding='utf-8'));assert sha(B/'model/model.safetensors')==config['model_sha256']
start=time.perf_counter();detector=XLMRDetector(B/'model',device='cpu',threads=8);load=time.perf_counter()-start
detector.score_texts(test.text.head(8).tolist(),8)
pred=test.drop(columns='text').copy();pred['logit']=np.nan;pred['windows']=0;timings=[];wall0=time.perf_counter()
for lang in ['ru','en','de','fr','zh']:
 part=test[test.lang.eq(lang)].sample(frac=1,random_state=202611119);durations=[];processed=0
 for start in range(0,len(part),128):
  batch=part.iloc[start:start+128];tic=time.perf_counter();z,w=detector.score_texts(batch.text.tolist(),8);elapsed=time.perf_counter()-tic
  durations.append(elapsed);pred.loc[batch.index,'logit']=z;pred.loc[batch.index,'windows']=w;processed+=len(batch)
  (OUT/'progress.json').write_text(json.dumps(dict(lang=lang,processed=processed,total=len(part),elapsed_s=time.perf_counter()-wall0)),encoding='utf-8')
 seconds=sum(durations);timings.append(dict(lang=lang,n=len(part),seconds=seconds,ms_per_doc=1000*seconds/len(part),docs_per_second=len(part)/seconds))
 print('CPU_COMPLETE',json.dumps(timings[-1]),flush=True)
pred['threshold']=pred.lang.map({l:r['logit_threshold'] for l,r in config['languages'].items()});pred['flag']=pred.logit.ge(pred.threshold)
gpu=ref.loc[pred.id,'logit_balanced5_frozen'].to_numpy();flags=ref.loc[pred.id,'flag_balanced5_frozen'].to_numpy()
mismatch=int((pred.flag.to_numpy()!=flags).sum());delta=float(np.max(np.abs(pred.logit-gpu)))
assert mismatch==0,(mismatch,delta)
pred.to_parquet(B/'evaluation/cpu_predictions.parquet',index=False)
def metric(part):
 y=part.label.to_numpy();z=part.logit.to_numpy();flags=part.flag.to_numpy();h=y==0;a=y==1
 pp,rr,_=precision_recall_curve(y,z)
 return dict(n=len(part),human=int(h.sum()),ai=int(a.sum()),tp=int(flags[a].sum()),fp=int(flags[h].sum()),recall=float(flags[a].mean()),fpr=float(flags[h].mean()),
  roc_auc=float(roc_auc_score(y,z)),pr_auc=float(auc(rr,pp)),ap=float(average_precision_score(y,z)))
pooled=[];bygen=[]
for row in timings:
 part=pred[pred.lang.eq(row['lang'])];pooled.append(dict(lang=row['lang'],**metric(part),cpu_ms_per_doc=row['ms_per_doc']))
 for gen in ['qwen3','gemma4']:bygen.append(dict(lang=row['lang'],generator=gen,**metric(part[part.generator.isin(['human',gen])])))
pd.DataFrame(pooled).to_csv(B/'evaluation/metrics_by_language.csv',index=False);pd.DataFrame(bygen).to_csv(B/'evaluation/metrics_by_generator.csv',index=False)
pd.DataFrame(timings).to_csv(B/'evaluation/cpu_speed.csv',index=False)
elapsed=sum(t['seconds'] for t in timings)
report=dict(n=len(test),threads=8,batch_size=8,warm=True,tokenization_included=True,model_load_seconds=load,total_inference_seconds=elapsed,
 ms_per_document=1000*elapsed/len(test),documents_per_second=len(test)/elapsed,device='cpu',torch=torch.__version__,platform=platform.platform(),
 hardware='Intel Core i7-14700KF',test_sha256=sha(data),model_sha256=config['model_sha256'],thresholds_sha256=sha(B/'model/thresholds.json'),
 decision_mismatches_vs_gpu=mismatch,max_logit_delta_vs_gpu=delta,per_language=timings,
 interpretation='Single full CPU pass over all 4213 retained independent test texts, batch throughput per document; not one-core latency. Same prior test, no fresh held-out confirmation after choosing release.')
(B/'evaluation/full_cpu_benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
samples={'ru':'Сегодня я забрал книгу из библиотеки и прочитал первую главу.',
 'en':'I collected the book from the library today and read the first chapter.',
 'de':'Heute habe ich das Buch aus der Bibliothek abgeholt und das erste Kapitel gelesen.',
 'fr':'J’ai récupéré le livre à la bibliothèque et lu le premier chapitre.',
 'zh':'今天我从图书馆借了一本书，回家后读完了第一章。'}
fixtures=[]
for lang,text in samples.items():
 r=detector.check(text,lang,8);fixtures.append(dict(lang=lang,text=text,logit=r['logit'],decision=r['decision']))
(B/'self_test_cases.json').write_text(json.dumps(fixtures,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'benchmark_complete.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(pooled).round(6).to_string(index=False),flush=True)
