from pathlib import Path
import argparse,json,sys
from detector import XLMRDetector
p=argparse.ArgumentParser(description="Multilingual AI-text detector, CPU")
p.add_argument('--lang',required=True,choices=['ru','en','de','fr','zh'])
p.add_argument('--file',type=Path,help='UTF-8 input file; otherwise read stdin')
p.add_argument('--threads',type=int,default=8)
args=p.parse_args()
text=args.file.read_text(encoding='utf-8-sig') if args.file else sys.stdin.read()
model=XLMRDetector(Path(__file__).resolve().parent/'model',device='cpu',threads=args.threads)
print(json.dumps(model.check(text,args.lang,batch_size=8),ensure_ascii=False,indent=2))
