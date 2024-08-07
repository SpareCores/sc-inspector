FROM ghcr.io/sparecores/sc-runner:main AS base
ENV REPO_URL="https://github.com/SpareCores/sc-inspector-data"
ENV REPO_PATH="/repo/sc-inspector-data"
ENV VIRTUAL_ENV="/venv/inspector"
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
ENV GIT_AUTHOR_EMAIL="inspector@sparecores.com"
ENV GIT_COMMITTER_EMAIL="inspector@sparecores.com"
ENV GIT_COMMITTER_NAME="Spare Cores"
RUN mkdir /repo
RUN \
    git config --global --add safe.directory /repo/sc-inspector-data && \
    git config --global user.email "inspector@sparecores.com" && \
    git config --global user.name "Spare Cores"
FROM base AS build
RUN \
    python -m venv --without-pip ${VIRTUAL_ENV} && \
    curl -sSLf https://bootstrap.pypa.io/get-pip.py | python -
ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

FROM base AS final
COPY --from=build ${VIRTUAL_ENV} ${VIRTUAL_ENV}
ADD inspector /inspector
ENTRYPOINT ["python", "/inspector/inspector.py"]