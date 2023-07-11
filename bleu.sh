#!/usr/bin/env bash 
REF_DIR=$1 
MT_DIR=$2
LOG_DIR=$3
echo "bleu score:" >> ${LOG_DIR}
perl multi-bleu.perl -lc ${REF_DIR} < ${MT_DIR} >> ${LOG_DIR}
rm ${REF_DIR}