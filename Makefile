# Makefile for local testing
AWS_ACCESS_KEY_ID ?= AKIAVRUVQPYA7PR7KSGP
AWS_SECRET_ACCESS_KEY ?= $(shell aws secretsmanager get-secret-value --secret-id aws/access_key/$(AWS_ACCESS_KEY_ID) --profile sc --region us-east-1 --query SecretString --output text)
AWS_DEFAULT_REGION ?= us-east-1
PULUMI_BACKEND_URL ?= s3://sc-inspector-data-pulumi-state-huco9rie

IMAGE_NAME ?= ghcr.io/sparecores/sc-inspector:main
SC_INSPECTOR_DATA_PATH ?= $$(realpath ../sc-inspector-data)
COMMON_OPTS = -w /app --privileged \
	-v /var/run/docker.sock:/var/run/docker.sock \
	-v $(PWD)/inspector:/app \
	-e AWS_ACCESS_KEY_ID \
	-e AWS_SECRET_ACCESS_KEY \
	-e AWS_DEFAULT_REGION \
	-e PULUMI_BACKEND_URL \
	--network=host
# -v $(SC_INSPECTOR_DATA_PATH):/repo/sc-inspector-data \

export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION
export PULUMI_BACKEND_URL

build:
	@docker build -t $(IMAGE_NAME) .

run:
	@docker run --rm -ti $(COMMON_OPTS) $(IMAGE_NAME)

shell:
	@docker run --rm -ti $(COMMON_OPTS) -m 4G --entrypoint bash $(IMAGE_NAME)
