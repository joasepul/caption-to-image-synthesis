FROM tensorflow/tensorflow:1.15.2-gpu-py3

RUN pip install opencv-python-headless nltk matplotlib gensim 
RUN pip install pandas==0.24.2

COPY *.py /

CMD ["bash"]
