FROM python:3.8

ARG NLTK_DATA=/nltk
ARG SRC_DIR=/src
ARG LOG_DIR=/logs
ARG PYTORCH_TRANSFORMERS_CACHE=/models

RUN mkdir -p ${NLTK_DATA} && mkdir -p ${SRC_DIR} && mkdir -p ${LOG_DIR} && mkdir -p ${PYTORCH_TRANSFORMERS_CACHE}
RUN touch ${LOG_DIR}/log.txt && chmod 777 ${LOG_DIR}/log.txt
# NLTK, spacy and gunicorn - separate setup
RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir nltk spacy
RUN python -m nltk.downloader punkt stopwords averaged_perceptron_tagger wordnet omw-1.4 -d $NLTK_DATA
ENV NLTK_DATA=$NLTK_DATA
# Other requirements
COPY requirements.txt $SRC_DIR
RUN pip3 install --default-timeout=1000 --no-cache-dir -r $SRC_DIR/requirements.txt

COPY --chmod=777 ./linking $SRC_DIR/linking
COPY --chmod=777 ./scripts $SRC_DIR/scripts
COPY --chmod=777 ./ptlm_wsid $SRC_DIR/ptlm_wsid
ENV PYTHONPATH "${PYTHONPATH}:${SRC_DIR}"
ENV SRC_DIR=$SRC_DIR
WORKDIR $SRC_DIR

ENTRYPOINT python $SRC_DIR/scripts/wsid_example.py
#ENTRYPOINT python $SRC_DIR/scripts/chi_example.py