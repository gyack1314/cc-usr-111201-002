"""展示已有的资料基本统计与离线策略价值分析，不改变原文件。"""
import argparse
import json
from .io import load
from .domain import summarize
from .ope import report
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inspect','ope']);p.add_argument('file');a=p.parse_args()
    value,digest=load(a.file)
    out={'source_sha256':digest,'summary':summarize(value)} if a.command=='inspect' else report(value,digest,source=a.file)
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
