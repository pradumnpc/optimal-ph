FROM continuumio/miniconda3

WORKDIR /home/biolib

RUN conda install --yes pytorch scikit-learn pandas numpy biopython \
    && conda clean -afy


COPY . .

ENTRYPOINT [ "python", "src/predict.py" ]
