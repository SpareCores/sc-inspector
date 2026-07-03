#!/bin/sh -x

# Redirect all output to /var/log/user_data.log for debugging
exec >> /var/log/user_data.log 2>&1

upload_user_data_log() {
    url="{LOG_UPLOAD_URL}"
    if [ -z "$url" ]; then
        return
    fi
    curl -sfS -X PUT -T /var/log/user_data.log -H "Content-Type: text/plain" "$url" || true
}

upload_run_status() {
    success="$1"
    exit_code="$2"
    url="{RUN_UPLOAD_URL}"
    if [ -z "$url" ]; then
        return
    fi
    terminated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    cat <<EOF | curl -sfS -X PUT -H "Content-Type: application/json" --data-binary @- "$url"
{"vendor":"{VENDOR}","instance":"{INSTANCE}","region":"{REGION}","zone":"{ZONE}","workflow":"{GITHUB_WORKFLOW}","run_id":"{GITHUB_RUN_ID}","terminated_at":"$terminated_at","success":$success,"exit_code":$exit_code}
EOF
}

finish_user_data() {
    if [ -n "${USER_DATA_FINISHED:-}" ]; then
        return
    fi
    USER_DATA_FINISHED=1
    exit_code="$1"
    if [ "$exit_code" -eq 0 ]; then
        success=true
    else
        success=false
    fi
    upload_user_data_log
    upload_run_status "$success" "$exit_code"
}

trap 'exit_code=$?; finish_user_data "$exit_code"' EXIT

TIMING_HOST_DIR="{HOST_TIMING_DIR}"
mkdir -p "$TIMING_HOST_DIR"
date -u +%Y-%m-%dT%H:%M:%SZ > "$TIMING_HOST_DIR/user_data_start"

# just to be sure, schedule a shutdown early
shutdown --no-wall +{SHUTDOWN_MINS}

# Disable GCP workload certificate refresh job (prevents recurring systemd starts).
# Do this early so it can't run during long package install steps below.
systemctl stop gce-workload-cert-refresh.service 2>/dev/null || true
systemctl disable gce-workload-cert-refresh.service 2>/dev/null || true
systemctl mask gce-workload-cert-refresh.service 2>/dev/null || true
# Some images schedule this via a timer unit.
systemctl stop gce-workload-cert-refresh.timer 2>/dev/null || true
systemctl disable gce-workload-cert-refresh.timer 2>/dev/null || true

export DEBIAN_FRONTEND=noninteractive
. /etc/os-release

# Retry apt-get update/install on transient mirror errors (hash sum mismatch, etc.)
# for up to 3 minutes with random backoff between attempts.
apt_retry() {
    deadline=$(($(date +%s) + 180))
    attempt=0
    while :; do
        attempt=$((attempt + 1))
        if apt-get "$@"; then
            return 0
        fi
        rc=$?
        now=$(date +%s)
        if [ "$now" -ge "$deadline" ]; then
            echo "apt_retry: failed after ${attempt} attempt(s), deadline exceeded: apt-get $*"
            return "$rc"
        fi
        remaining=$((deadline - now))
        jitter=$(awk 'BEGIN{srand(); print int(5 + rand() * 26)}')
        [ "$jitter" -gt "$remaining" ] && jitter="$remaining"
        [ "$jitter" -lt 1 ] && jitter=1
        echo "apt_retry: attempt ${attempt} failed (apt-get $*), sleeping ${jitter}s before retry"
        sleep "$jitter"
        case "$1" in
            install)
                apt-get clean || true
                apt-get update -y || true
                ;;
        esac
    done
}

apt_retry update -y
# Add the required repositories to Apt sources:
apt_retry install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings

DOCKER_REPO_DEFAULT="https://download.docker.com/linux/$ID"
DOCKER_REPO_ALIYUN="https://mirrors.aliyun.com/docker-ce/linux/$ID"
DOCKER_APT_PKGS="docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin openssh-client"
DOCKER_REPO="$DOCKER_REPO_DEFAULT"

setup_docker_repo() {
    repo="$1"
    echo "Setting up Docker apt repo: $repo"
    curl -fsSL "$repo/gpg" -o /etc/apt/keyrings/docker.asc || return 1
    chmod a+r /etc/apt/keyrings/docker.asc
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] $repo \
      $VERSION_CODENAME stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null
}

if ! setup_docker_repo "$DOCKER_REPO"; then
    if [ "{VENDOR}" = "alicloud" ]; then
        echo "download.docker.com unreachable, falling back to Aliyun mirror"
        DOCKER_REPO="$DOCKER_REPO_ALIYUN"
        setup_docker_repo "$DOCKER_REPO" || exit 1
    else
        exit 1
    fi
fi

install_docker_apt() {
    apt_retry update -y && apt_retry install -y $NVIDIA_PKGS $DOCKER_APT_PKGS
}
# nvidia drivers/toolkit when GPU_COUNT != 0
NVIDIA_PKGS=""
NVIDIA_DRIVER_PKG=""
ALIYUN_DRIVER_PLUGIN=""
FRACTIONAL_GPU_DRIVER=""

if [ "{GPU_COUNT}" != "0" ] && [ "{GPU_COUNT}" != "0.0" ]; then
    # Ubuntu cloud images (e.g. UpCloud) may boot with nouveau bound to the GPU.
    # Blacklist and unbind without reboot so the proprietary driver can probe the device.
    disable_nouveau() {
        cat > /etc/modprobe.d/nouveau.conf <<'EOF'
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
EOF
    }
    # Headless cloud GPUs: apt nvidia-driver sets modeset=1 and 71-nvidia.rules auto-loads
    # nvidia-drm. With no display connectors the DRM driver unloads immediately; udev then
    # reloads it (~1 Hz), causing modprobe/nvidia-modeset CPU spin and PCI bind/unbind churn.
    configure_headless_nvidia() {
        cat > /etc/modprobe.d/nvidia-graphics-drivers-kms.conf <<'EOF'
options nvidia_drm modeset=0
options nvidia NVreg_PreserveVideoMemoryAllocations=1
options nvidia NVreg_TemporaryFilePath=/var
EOF
        cat > /etc/udev/rules.d/72-nvidia-disable-runtime-pm.rules <<'EOF'
ACTION=="add|bind", SUBSYSTEM=="pci", ATTR{vendor}=="0x10de", ATTR{class}=="0x03[0-9]*", TEST=="power/control", ATTR{power/control}="on"
EOF
        cat > /etc/udev/rules.d/73-nvidia-headless-benchmark.rules <<'EOF'
ACTION=="add", DEVPATH=="/bus/pci/drivers/nvidia", RUN-="/sbin/modprobe nvidia-modeset"
ACTION=="remove", DEVPATH=="/bus/pci/drivers/nvidia", RUN-="/sbin/modprobe -r nvidia-modeset"
ACTION=="add", DEVPATH=="/bus/pci/drivers/nvidia", RUN-="/sbin/modprobe nvidia-drm"
ACTION=="remove", DEVPATH=="/bus/pci/drivers/nvidia", RUN-="/sbin/modprobe -r nvidia-drm"
EOF
        udevadm control --reload-rules 2>/dev/null || true
    }
    # Ubuntu transitional metapackages (e.g. nvidia-driver-590-open) depend on the real
    # driver package (nvidia-driver-595-open). Resolve before pruning other drivers.
    resolve_nvidia_driver_pkg() {
        pkg="$1"
        while true; do
            dep=$(apt-cache show "$pkg" 2>/dev/null | awk -F': ' '/^Depends:/{print $2}' | tr ',' '\n' | sed 's/^[[:space:]]*//' | grep -E '^nvidia-driver-[0-9]' | head -1)
            [ -n "$dep" ] && [ "$dep" != "$pkg" ] || break
            pkg="$dep"
        done
        echo "$pkg"
    }
    activate_nvidia_driver() {
        disable_nouveau
        for pci_addr in $(lspci -d 10de: -n 2>/dev/null | awk '{print "0000:"$1}'); do
            [ -d "/sys/bus/pci/devices/$pci_addr/driver" ] || continue
            driver=$(basename "$(readlink "/sys/bus/pci/devices/$pci_addr/driver" 2>/dev/null)" 2>/dev/null || echo "")
            if [ "$driver" = "nouveau" ]; then
                echo "$pci_addr" > "/sys/bus/pci/devices/$pci_addr/driver/unbind" 2>/dev/null || true
            fi
        done
        modprobe -r nvidia_drm nvidia_modeset nvidia 2>/dev/null || true
        modprobe -r nouveau drm_ttm_helper ttm drm_gpuvm drm_exec gpu_sched drm_display_helper drm_kms_helper drm 2>/dev/null || true
        modprobe -r nouveau 2>/dev/null || true
        modprobe nvidia
        modprobe nvidia_uvm 2>/dev/null || true
    }
    # nvidia container toolkit (always for GPU instances)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    # Check if this is a fractional GPU instance (GPU_COUNT is not a whole number)
    # Use awk to check if the number has a non-zero fractional part
    IS_FRACTIONAL=$(echo "{GPU_COUNT}" | awk '{if ($1 != int($1)) print "true"; else print "false"}')
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
                apt_retry install -y build-essential
                # Download and install NVIDIA driver from AWS S3
                aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ /tmp/ --no-sign-request
                chmod +x /tmp/NVIDIA-Linux-x86_64*.run
                /tmp/NVIDIA-Linux-x86_64*.run -s --no-drm
                FRACTIONAL_GPU_DRIVER="aws-s3"
                NVIDIA_PKGS="nvidia-container-toolkit"
                ;;
            azure)
                # Azure fractional GPUs: use NVIDIA GRID drivers for NV-series instances
                # Reference: https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup
                # Fractional GPU instances: NV4as_v4, NV8as_v4, NV16as_v4, NV6s_v2, NV4ads_V710_v5, NV8ads_V710_v5, 
                # NV12ads_V710_v5, NV6ads_A10_v5, NV12ads_A10_v5, NV18ads_A10_v5
                case "{INSTANCE}" in
                    *NV*as_v4|*NV*s_v2|*NV*ads_V710_v5|*NV*ads_A10_v5)
                        # Install GRID driver for NV-series fractional GPU instances
                        # Install build essentials and kernel headers
                        apt_retry install -y build-essential linux-azure
                        
                        # Disable nouveau driver (incompatible with NVIDIA)
                        cat > /etc/modprobe.d/nouveau.conf <<EOF
blacklist nouveau
blacklist lbm-nouveau
EOF
                        # Download and install GRID driver from Microsoft
                        wget -O /tmp/NVIDIA-Linux-x86_64-grid.run https://go.microsoft.com/fwlink/?linkid=874272
                        chmod +x /tmp/NVIDIA-Linux-x86_64-grid.run
                        /tmp/NVIDIA-Linux-x86_64-grid.run -s --no-drm
                        
                        # Configure GRID licensing
                        if [ -f /etc/nvidia/gridd.conf.template ]; then
                            cp /etc/nvidia/gridd.conf.template /etc/nvidia/gridd.conf
                            # Add required configuration
                            if ! grep -q "IgnoreSP=FALSE" /etc/nvidia/gridd.conf; then
                                echo "IgnoreSP=FALSE" >> /etc/nvidia/gridd.conf
                            fi
                            if ! grep -q "EnableUI=FALSE" /etc/nvidia/gridd.conf; then
                                echo "EnableUI=FALSE" >> /etc/nvidia/gridd.conf
                            fi
                            # Remove FeatureType if present
                            sed -i '/^FeatureType=0/d' /etc/nvidia/gridd.conf
                        fi
                        FRACTIONAL_GPU_DRIVER="azure-grid"
                        NVIDIA_PKGS="nvidia-container-toolkit"
                        ;;
                    *)
                        # Non-GRID instances: use standard driver detection
                        ;;
                esac
                ;;
            *)
                # Other vendors: use standard driver detection
                # Fall through to standard detection logic
                ;;
        esac
    fi

    # Alibaba vGPU VWS types need GRID via acs-plugin-manager regardless of GPU_COUNT fractionality
    if [ "{VENDOR}" = "alicloud" ] && [ -z "$FRACTIONAL_GPU_DRIVER" ]; then
        if command -v acs-plugin-manager >/dev/null 2>&1; then
            case "{INSTANCE}" in
                *-vws*)     PLUGIN="grid_driver_install" ;;
                *sgn8ia*)   PLUGIN="gpu_grid_driver_install" ;;
                *)          PLUGIN="" ;;
            esac
            if [ -n "$PLUGIN" ] && acs-plugin-manager --list 2>/dev/null | grep -q "$PLUGIN"; then
                ALIYUN_DRIVER_PLUGIN="$PLUGIN"
                FRACTIONAL_GPU_DRIVER="alicloud-acs"
                NVIDIA_PKGS="nvidia-container-toolkit"
            fi
        fi
    fi
    
    # Detect GPU architecture and select appropriate driver (only if not using fractional GPU driver)
    if [ -z "$FRACTIONAL_GPU_DRIVER" ]; then
        # Only proceed if NVIDIA_PKGS hasn't been set by Alibaba plugin or other means
        if [ -z "$NVIDIA_PKGS" ]; then
            disable_nouveau
            apt_retry install -y jq lshw software-properties-common
            add-apt-repository ppa:graphics-drivers/ppa -y
            
            # Detect GPU using lshw (more reliable than lspci for product names)
            # Default to open driver for modern GPUs (Turing+, Ada, Hopper, Blackwell)
            # Open driver is required for Blackwell/Grace Hopper and recommended for Ada/Hopper/Ampere/Turing
            DRIVER_VARIANT="open"
            DRIVER_VERSION="590"
            
            # Check if lshw and jq are available and get GPU info
            if command -v lshw >/dev/null 2>&1 && command -v jq >/dev/null 2>&1; then
                # Get NVIDIA GPU product names from lshw JSON output using jq
                GPU_INFO=$(lshw -c display -json 2>/dev/null | jq -r '.. | objects | select(.vendor? == "NVIDIA Corporation") | .product' 2>/dev/null || echo "")
                
                # Check for older architectures that need proprietary server driver
                # Volta (GV1xx, V100, V100S), Pascal (GP1xx, P100, P40, P4), Maxwell (GM1xx, M60, M40), Kepler (GK1xx, K80, K40)
                if echo "$GPU_INFO" | grep -qiE 'GV1[0-9]{2}|V100S?|GP1[0-9]{2}|P[146]0|GM1[0-9]{2}|M[46]0|GK1[0-9]{2}|K[248]0'; then
                    DRIVER_VARIANT="server"
                    DRIVER_VERSION="580"
                fi
            fi
            
            NVIDIA_DRIVER_PKG="nvidia-driver-$DRIVER_VERSION-$DRIVER_VARIANT"
            NVIDIA_PKGS="$NVIDIA_DRIVER_PKG nvidia-container-toolkit"
        fi
    fi
fi
if ! install_docker_apt; then
    if [ "{VENDOR}" = "alicloud" ] && [ "$DOCKER_REPO" != "$DOCKER_REPO_ALIYUN" ]; then
        echo "Docker apt install failed on download.docker.com, falling back to Aliyun mirror"
        DOCKER_REPO="$DOCKER_REPO_ALIYUN"
        setup_docker_repo "$DOCKER_REPO" || exit 1
        apt-get clean || true
        install_docker_apt || exit 1
    else
        exit 1
    fi
fi
# install Alibaba GPU driver via acs-plugin-manager when applicable (never fail the script)
if [ -n "$ALIYUN_DRIVER_PLUGIN" ]; then
    acs-plugin-manager --remove --plugin "$ALIYUN_DRIVER_PLUGIN" >/dev/null 2>&1 || true
    acs-plugin-manager --exec --plugin "$ALIYUN_DRIVER_PLUGIN" >/dev/null 2>&1 || true
fi
# Load the apt-installed NVIDIA driver without reboot (UpCloud and similar Ubuntu images)
if [ -n "$NVIDIA_DRIVER_PKG" ]; then
    NVIDIA_DRIVER_PKG=$(resolve_nvidia_driver_pkg "$NVIDIA_DRIVER_PKG")
    for pkg in $(dpkg-query -W -f='${Package}\n' 'nvidia-driver-*' 2>/dev/null); do
        [ "$pkg" = "$NVIDIA_DRIVER_PKG" ] && continue
        apt-get remove -y "$pkg" || true
    done
    configure_headless_nvidia
    activate_nvidia_driver
fi
if [ "{GPU_COUNT}" != "0" ] && [ "{GPU_COUNT}" != "0.0" ]; then
    nvidia-ctk runtime configure --runtime=nvidia 2>/dev/null || true
fi
# Docker defaults to 3 concurrent layer downloads per pull; raise for better aggregate
# bandwidth on high-latency links (e.g. mainland China -> ghcr.io).
mkdir -p /etc/docker
if [ "{VENDOR}" = "alicloud" ]; then
    # registry-1.docker.io is unreachable from Alibaba Cloud; mirror Hub for nvidia/cuda etc.
    # Community mirrors (Jun 2026); Docker tries in order. Skip dead legacy mirrors
    # (registry.cn-hangzhou.aliyuncs.com, hub-mirror.c.163.com) and rate-limited ones.
    cat > /etc/docker/daemon.json <<'EOF'
{
  "max-concurrent-downloads": 20,
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://docker.1ms.run",
    "https://hub.rat.dev",
    "https://docker.1panel.live"
  ]
}
EOF
else
    cat > /etc/docker/daemon.json <<'EOF'
{
  "max-concurrent-downloads": 20
}
EOF
fi
systemctl restart docker
# set up SSH for git operations
mkdir -p /root/.ssh
chmod 700 /root/.ssh
echo "{SSH_DEPLOY_KEY_B64}" | base64 -d > /root/.ssh/id_rsa
chmod 600 /root/.ssh/id_rsa
ssh-keyscan github.com >> /root/.ssh/known_hosts
# stop some services to preserve memory and reduce interference with benchmarks
snap stop amazon-ssm-agent
systemctl stop chrony acpid fwupd cron multipathd snapd systemd-timedated google-osconfig-agent google-guest-agent \
    networkd-dispatcher unattended-upgrades polkit packagekit systemd-udevd hv-kvp-daemon.service \
    cloud-init cloud-config cloud-final cloud-init-local \
    aegis aliyun AssistDaemon tuned rsyslog
systemctl disable aegis aliyun AssistDaemon tuned rsyslog
# stop Alicloud aegis security agent processes directly (they may respawn)
pkill -9 -f AliYunDun
pkill -9 -f aegis
pkill -9 -f aliyun-service
pkill -9 -f assist_daemon
# disable motd-news (makes network calls on login)
sed -i 's/ENABLED=1/ENABLED=0/' /etc/default/motd-news 2>/dev/null
chmod -x /etc/update-motd.d/* 2>/dev/null
# remove unwanted packages (keep NVIDIA libs; autoremove would drop them after metapackage pruning)
if [ -n "$NVIDIA_DRIVER_PKG" ]; then
    for pkg in $(dpkg-query -W -f='${Package}\n' 'nvidia-*' 'libnvidia-*' 2>/dev/null); do
        apt-mark manual "$pkg" 2>/dev/null || true
    done
fi
apt-get autoremove -y $(dpkg-query -W -f='${Package}\n' \
    apport fwupd unattended-upgrades snapd packagekit \
    walinuxagent google-osconfig-agent 2>/dev/null)
# https://github.com/NVIDIA/nvidia-container-toolkit/issues/202
# on some machines docker initialization times out with a lot of GPUs. Enable persistence mode to overcome that.
nvidia-smi -pm 1
date -u +%Y-%m-%dT%H:%M:%SZ > "$TIMING_HOST_DIR/user_data_end"
TRACKER_DIR=/opt/sparecores-inspector/resource-tracker
mkdir -p "$TRACKER_DIR"
if [ ! -x "$TRACKER_DIR/resource-tracker" ]; then
    cid=$(docker create --entrypoint /usr/local/bin/resource-tracker ghcr.io/sparecores/resource-tracker:main /bin/true)
    docker cp "$cid:/usr/local/bin/resource-tracker" "$TRACKER_DIR/resource-tracker"
    docker rm "$cid"
    chmod 755 "$TRACKER_DIR/resource-tracker"
fi
if [ ! -x "$TRACKER_DIR/resource-tracker" ]; then
    echo "resource-tracker staging failed: $TRACKER_DIR/resource-tracker missing" >&2
    exit 1
fi
set +e
if [ "{INSPECTOR_ROLE}" = "client" ]; then
    BENCHMARK_OUTPUT_DIR=/var/lib/sparecores-inspector/benchmark-output
    mkdir -p "$BENCHMARK_OUTPUT_DIR"
    docker run --rm --network=host --privileged -v /var/run/docker.sock:/var/run/docker.sock \
        -v "$BENCHMARK_OUTPUT_DIR:/benchmark-output" \
        -e VENDOR={VENDOR} -e INSTANCE={INSTANCE} \
        -e MP_AUTHKEY_B64={MP_AUTHKEY_B64} -e MP_PORT={MP_PORT} \
        -e HOST_BENCHMARK_OUTPUT_DIR="$BENCHMARK_OUTPUT_DIR" \
        -e BENCHMARK_OUTPUT_MOUNT=/benchmark-output \
        -e GITHUB_RUN_ID={GITHUB_RUN_ID} -e SENTINEL_API_TOKEN={SENTINEL_API_TOKEN} \
        -e HF_TOKEN={HF_TOKEN} \
        ghcr.io/sparecores/sc-inspector:main companion --vendor {VENDOR} --instance {INSTANCE} --listen-port {MP_PORT}
    inspect_exit=$?
else
docker run --rm --network=host --privileged -v /var/run/docker.sock:/var/run/docker.sock -v /root/.ssh:/root/.ssh \
    -v "$TIMING_HOST_DIR:/host-timing:ro" \
    -e REPO_URL={REPO_URL} \
    -e GITHUB_SERVER_URL={GITHUB_SERVER_URL} \
    -e GITHUB_REPOSITORY={GITHUB_REPOSITORY} \
    -e GITHUB_RUN_ID={GITHUB_RUN_ID} \
    -e HOST_TIMING_DIR="$TIMING_HOST_DIR" \
    -e BENCHMARK_SECRETS_PASSPHRASE={BENCHMARK_SECRETS_PASSPHRASE} \
    -e SENTINEL_API_TOKEN={SENTINEL_API_TOKEN} \
    -e HF_TOKEN={HF_TOKEN} \
    -e TASK_LOGS_S3_POST_B64={TASK_LOGS_S3_POST_B64} \
    -e MP_AUTHKEY_B64={MP_AUTHKEY_B64} -e MP_PORT={MP_PORT} \
    -e CLIENT_PRIVATE_IP={CLIENT_PRIVATE_IP} \
    -e MULTI_VM_CLIENT_INSTANCE={MULTI_VM_CLIENT_INSTANCE} \
    -e MULTI_VM_CLIENT_CPU_ARCH={MULTI_VM_CLIENT_CPU_ARCH} \
    -e MULTI_VM_CLIENT_VCPUS={MULTI_VM_CLIENT_VCPUS} \
    -e PROVISIONED_DISK_GIB={PROVISIONED_DISK_GIB} \
    -e CLIENT_DISK_GIB={CLIENT_DISK_GIB} \
    -e HOST_RESOURCE_TRACKER_DIR="$TRACKER_DIR" \
    ghcr.io/sparecores/sc-inspector:main inspect --vendor {VENDOR} --instance {INSTANCE} --gpu-count {GPU_COUNT}
inspect_exit=$?
fi
set -e
finish_user_data "$inspect_exit"
poweroff
