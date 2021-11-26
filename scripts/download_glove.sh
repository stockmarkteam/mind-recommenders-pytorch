#!/bin/bash
wget https://nlp.stanford.edu/data/glove.840B.300d.zip -O models/glove.zip
unzip models/glove.zip -d models/glove/
rm models/glove.zip
