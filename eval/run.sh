#!/bin/bash
python format_line_answer.py --type full --model codegm --tp basic &&
python format_line_answer.py --type full --model codegm --tp basic2 &&
python format_line_answer.py --type full --model codegm --tp cot &&
python format_line_answer.py --type full --model codegm --tp cot2 &&
python format_line_answer.py --type full --model dscoder --tp basic &&
python format_line_answer.py --type full --model dscoder --tp basic2