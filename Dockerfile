# SC_RUNNER_IMAGE_TAG selects the sc-runner base image (main for dev, vX.Y.Z on release).
ARG SC_RUNNER_IMAGE_TAG=main
FROM ghcr.io/sparecores/sc-runner:${SC_RUNNER_IMAGE_TAG} AS base
ENV REPO_URL="https://github.com/SpareCores/sc-inspector-data"
ENV REPO_PATH="/repo/sc-inspector-data"
ENV VIRTUAL_ENV="/venv/inspector"
ENV PATH=/root/.local/bin:${VIRTUAL_ENV}/bin:${PATH}
ENV GIT_AUTHOR_EMAIL="inspector@sparecores.com"
ENV GIT_COMMITTER_EMAIL="inspector@sparecores.com"
ENV GIT_COMMITTER_NAME="Spare Cores"
RUN mkdir /repo
RUN \
    git config --global --add safe.directory /repo/sc-inspector-data && \
    git config --global user.email "inspector@sparecores.com" && \
    git config --global user.name "Spare Cores" && \
    git config --global core.bigFileThreshold 1 && \
    git config --global core.deltaBaseCacheLimit 0 && \
    git config --global gc.auto 0 && \
    git config --global pack.deltaCacheLimit 0 && \
    git config --global pack.deltaCacheSize 1 && \
    git config --global pack.threads 1 && \
    git config --global pack.windowMemory 10m && \
    git config --global pack.packSizeLimit 20m && \
    git config --global core.packedGitWindowSize 16m && \
    git config --global checkout.thresholdForParallelism 99999999 && \
    git config --global core.compression 0 && \
    git config --global index.threads 1
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get update && \
    apt-get install -y lshw jq git-restore-mtime hwloc-nox

FROM base AS build
# SC_RUNNER_VERSION pins sparecores-runner in the inspector venv (release builds).
ARG SC_RUNNER_VERSION
ADD requirements.txt /tmp/requirements.txt
RUN \
    python -m venv --without-pip ${VIRTUAL_ENV} && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    if [ -n "$SC_RUNNER_VERSION" ]; then \
      sed -i "s/^sparecores-runner.*/sparecores-runner==${SC_RUNNER_VERSION}/" /tmp/requirements.txt; \
    fi && \
    uv pip install -r /tmp/requirements.txt

FROM base AS final
COPY --from=build ${VIRTUAL_ENV} ${VIRTUAL_ENV}
ADD inspector /inspector
ENTRYPOINT ["python", "/inspector/inspector.py"]
