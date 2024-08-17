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
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get update && \
    apt-get install -y lshw jq

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