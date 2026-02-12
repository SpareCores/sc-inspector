#!/bin/sh

# just to be sure, schedule a shutdown early
shutdown --no-wall +{SHUTDOWN_MINS}

export DEBIAN_FRONTEND=noninteractive
. /etc/os-release
apt-get update -y
# Add the required repositories to Apt sources:
apt-get install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
# docker
curl -fsSL https://download.docker.com/linux/$ID/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/$ID \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null
# nvidia drivers/toolkit when GPU_COUNT != 0
NVIDIA_PKGS=""
ALIYUN_DRIVER_PLUGIN=""
FRACTIONAL_GPU_DRIVER=""

if [ "{GPU_COUNT}" != "0" ] && [ "{GPU_COUNT}" != "0.0" ]; then
    # nvidia container toolkit (always for GPU instances)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    # Check if this is a fractional GPU instance (GPU_COUNT is not a whole number)
    # Use awk to check if the number has a non-zero fractional part
    IS_FRACTIONAL=$(echo "{GPU_COUNT}" | awk '{{if ($1 != int($1)) print "true"; else print "false"}}')
    if [ "$IS_FRACTIONAL" = "true" ]; then
        IS_FRACTIONAL_GPU=true
    else
        IS_FRACTIONAL_GPU=false
    fi
    
    # Handle fractional GPU driver installation per vendor
    if [ "$IS_FRACTIONAL_GPU" = "true" ]; then
        case "{VENDOR}" in
            aws)
                # AWS fractional GPUs: use NVIDIA drivers from S3
                # Install aws-cli before snap is removed
                snap install aws-cli --classic
                # Install build tools required for NVIDIA driver compilation
                apt-get install -y build-essential
                # Download and install NVIDIA driver from AWS S3
                aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ /tmp/ --no-sign-request
                chmod +x /tmp/NVIDIA-Linux-x86_64*.run
                /tmp/NVIDIA-Linux-x86_64*.run -s --no-drm
                FRACTIONAL_GPU_DRIVER="aws-s3"
                NVIDIA_PKGS="nvidia-container-toolkit"
                ;;
            gcp)
                # GCP fractional GPUs: use NVIDIA drivers from GCS or NVIDIA GRID drivers
                # GCP provides pre-installed drivers for some instances, check if already present
                if ! (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1); then
                    # Install NVIDIA GRID drivers for virtual workstation instances
                    # GCP provides these via their own repositories
                    curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-docker-keyring.gpg
                    # Use standard PPA as fallback for GCP
                    add-apt-repository ppa:graphics-drivers/ppa -y
                    NVIDIA_PKGS="nvidia-driver-550-server nvidia-container-toolkit"
                fi
                FRACTIONAL_GPU_DRIVER="gcp-grid"
                ;;
            azure)
                # Azure fractional GPUs: use NVIDIA GRID drivers for NVv4 and NVadsA10_v5 series
                # Azure provides NVIDIA drivers via extensions, but we install manually for consistency
                if ! (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1); then
                    add-apt-repository ppa:graphics-drivers/ppa -y
                    # Azure NVadsA10_v5 uses A10 GPUs, use open driver
                    NVIDIA_PKGS="nvidia-driver-550-open nvidia-container-toolkit"
                fi
                FRACTIONAL_GPU_DRIVER="azure-grid"
                ;;
            alicloud)
                # Alibaba Cloud fractional GPUs: handled by acs-plugin-manager below
                # Fall through to standard Alibaba handling
                ;;
            *)
                # Other vendors: use standard driver detection
                # Fall through to standard detection logic
                ;;
        esac
    fi
    
    # driver: use Alibaba acs-plugin-manager for eligible instance types, else nvidia PPA
    if [ -z "$FRACTIONAL_GPU_DRIVER" ] && [ "{VENDOR}" = "alicloud" ] && command -v acs-plugin-manager >/dev/null 2>&1; then
        case "{INSTANCE}" in
            *sgn7i-vws*) PLUGIN="grid_driver_install" ;;
            *sgn8ia*)    PLUGIN="gpu_grid_driver_install" ;;
            *)           PLUGIN="" ;;
        esac
        if [ -n "$PLUGIN" ] && acs-plugin-manager --list 2>/dev/null | grep -q "$PLUGIN"; then
            if ! (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1); then
                ALIYUN_DRIVER_PLUGIN="$PLUGIN"
            fi
            NVIDIA_PKGS="nvidia-container-toolkit"
        fi
    fi
    
    # Detect GPU architecture and select appropriate driver (only if not using fractional GPU driver)
    if [ -z "$FRACTIONAL_GPU_DRIVER" ]; then
        # Only proceed if NVIDIA_PKGS hasn't been set by Alibaba plugin or other means
        if [ -z "$NVIDIA_PKGS" ]; then
            add-apt-repository ppa:graphics-drivers/ppa -y
            
            # Detect GPU using lshw (more reliable than lspci for product names)
            # Default to open driver for modern GPUs (Turing+, Ada, Hopper, Blackwell)
            # Open driver is required for Blackwell/Grace Hopper and recommended for Ada/Hopper/Ampere/Turing
            DRIVER_VARIANT="open"
            
            # Check if lshw and jq are available and get GPU info
            if command -v lshw >/dev/null 2>&1 && command -v jq >/dev/null 2>&1; then
                # Get NVIDIA GPU product names from lshw JSON output using jq
                GPU_INFO=$(lshw -c display -json 2>/dev/null | jq -r '.. | objects | select(.vendor? == "NVIDIA Corporation") | .product' 2>/dev/null || echo "")
                
                # Check for older architectures that need proprietary server driver
                # Volta (GV1xx, V100, V100S), Pascal (GP1xx, P100, P40, P4), Maxwell (GM1xx, M60, M40), Kepler (GK1xx, K80, K40)
                if echo "$GPU_INFO" | grep -qiE 'GV1[0-9]{{2}}|V100S?|GP1[0-9]{{2}}|P[146]0|GM1[0-9]{{2}}|M[46]0|GK1[0-9]{{2}}|K[248]0'; then
                    DRIVER_VARIANT="server"
                fi
            fi
            
            NVIDIA_PKGS="nvidia-driver-590-$DRIVER_VARIANT nvidia-container-toolkit"
        fi
    fi
fi
apt-get update -y >> /tmp/output 2>&1
apt-get install -y $NVIDIA_PKGS docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin openssh-client >> /tmp/output 2>&1
# install Alibaba GPU driver via acs-plugin-manager when applicable (never fail the script)
if [ -n "$ALIYUN_DRIVER_PLUGIN" ]; then
    acs-plugin-manager --remove --plugin "$ALIYUN_DRIVER_PLUGIN" >/dev/null 2>&1 || true
    acs-plugin-manager --exec --plugin "$ALIYUN_DRIVER_PLUGIN" >/dev/null 2>&1 || true
fi
systemctl restart docker
# set up SSH for git operations
mkdir -p /root/.ssh
chmod 700 /root/.ssh
echo "{SSH_DEPLOY_KEY_B64}" | base64 -d > /root/.ssh/id_rsa
chmod 600 /root/.ssh/id_rsa
ssh-keyscan github.com >> /root/.ssh/known_hosts 2>>/tmp/output
# stop some services to preserve memory and reduce interference with benchmarks
snap stop amazon-ssm-agent >> /tmp/output 2>&1
systemctl stop chrony acpid fwupd cron multipathd snapd systemd-timedated google-osconfig-agent google-guest-agent \
    networkd-dispatcher unattended-upgrades polkit packagekit systemd-udevd hv-kvp-daemon.service \
    cloud-init cloud-config cloud-final cloud-init-local \
    aegis aliyun AssistDaemon tuned rsyslog >> /tmp/output 2>&1
systemctl disable aegis aliyun AssistDaemon tuned rsyslog >> /tmp/output 2>&1
# stop Alicloud aegis security agent processes directly (they may respawn)
pkill -9 -f AliYunDun >> /tmp/output 2>&1
pkill -9 -f aegis >> /tmp/output 2>&1
pkill -9 -f aliyun-service >> /tmp/output 2>&1
pkill -9 -f assist_daemon >> /tmp/output 2>&1
# disable motd-news (makes network calls on login)
sed -i 's/ENABLED=1/ENABLED=0/' /etc/default/motd-news 2>/dev/null
chmod -x /etc/update-motd.d/* 2>/dev/null
# remove unwanted packages
apt-get autoremove -y $(dpkg-query -W -f='${{Package}}\n' \
    apport fwupd unattended-upgrades snapd packagekit \
    walinuxagent google-osconfig-agent 2>/dev/null) >> /tmp/output 2>&1
# https://github.com/NVIDIA/nvidia-container-toolkit/issues/202
# on some machines docker initialization times out with a lot of GPUs. Enable persistence mode to overcome that.
nvidia-smi -pm 1
docker run --rm --network=host --privileged -v /var/run/docker.sock:/var/run/docker.sock -v /root/.ssh:/root/.ssh \
    -e REPO_URL={REPO_URL} \
    -e GITHUB_SERVER_URL={GITHUB_SERVER_URL} \
    -e GITHUB_REPOSITORY={GITHUB_REPOSITORY} \
    -e GITHUB_RUN_ID={GITHUB_RUN_ID} \
    -e BENCHMARK_SECRETS_PASSPHRASE={BENCHMARK_SECRETS_PASSPHRASE} \
    ghcr.io/sparecores/sc-inspector:main inspect --vendor {VENDOR} --instance {INSTANCE} --gpu-count {GPU_COUNT} >> /tmp/output 2>&1
poweroff
