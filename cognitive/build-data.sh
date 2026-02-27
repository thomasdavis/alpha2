#!/bin/bash
# Concatenate historic chat data + Gutenberg prose into one ~870MB training file
cat data/historic.txt data/concordance-v2.txt > cognitive/cognitive-v1.txt
echo "Built cognitive/cognitive-v1.txt ($(du -h cognitive/cognitive-v1.txt | cut -f1))"
