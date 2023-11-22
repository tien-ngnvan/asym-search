# Build stage:
FROM hieupth/mamba AS build

RUN mamba create --name symsearch python=3.9 openpyxl flask pytorch cpuonly transformers pypdf faiss-cpu langchain chromadb -c conda-forge -c pytorch -y
# ADD . . 
# ENTRYPOINT ["tail", "-f", "/dev/null"]

RUN mamba install -c conda-forge conda-pack
RUN conda-pack -n symsearch -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar

# Runtime stage:
FROM debian:buster AS runtime
# Copy /venv from the previous stage:
COPY --from=build /venv /venv
ADD . .
SHELL ["/bin/bash", "-c"]
EXPOSE 5000
ENTRYPOINT source /venv/bin/activate && python -m symsearch.demo