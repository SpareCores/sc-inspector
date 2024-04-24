# Makefile for local testing
IMAGE_NAME ?= sc-inspector
SC_INSPECTOR_DATA_PATH ?= $$(realpath ../sc-inspector-data)
COMMON_OPTS = -w /app --privileged \
	-v /var/run/docker.sock:/var/run/docker.sock \
	-v $(PWD)/inspector:/app \
	-v $(SC_INSPECTOR_DATA_PATH):/repo/sc-inspector-data

build:
	@docker build -t $(IMAGE_NAME) .

run:
	@docker run --rm -ti $(COMMON_OPTS) $(IMAGE_NAME)

shell:
	@docker run --rm -ti $(COMMON_OPTS) --entrypoint bash $(IMAGE_NAME)