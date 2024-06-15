#!/bin/bash
python infer/qwen_line.py --type qlora --tp basic &&
python infer/qwen_line.py --type qlora --tp basic2 &&
python infer/qwen_line.py --type qlora --tp cot &&
python infer/qwen_line.py --type qlora --tp cot2


