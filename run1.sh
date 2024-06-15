#!/bin/bash
python infer/qwen_line.py --type lora --tp cot &&
python infer/qwen_line.py --type lora --tp cot2 
