# RTX 6000 NVIDIA Container Deployment Guide

This guide provides step-by-step instructions for deploying the Math Learning application on a VM with NVIDIA RTX 6000 GPU support.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [VM Setup](#vm-setup)
3. [NVIDIA Driver Installation](#nvidia-driver-installation)
4. [Docker and NVIDIA Container Toolkit Setup](#docker-and-nvidia-container-toolkit-setup)
5. [Application Deployment](#application-deployment)
6. [GPU Verification](#gpu-verification)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements
- VM with NVIDIA RTX 6000 GPU (or GPU passthrough configured)
- Minimum 16GB RAM
- Minimum 50GB disk space
- Ubuntu 20.04/22.04 or RHEL 8/9

### Software Requirements
- Root or sudo access
- Internet connectivity
- SSH access to the VM

## VM Setup

### 1. Update System Packages

```bash
# For Ubuntu/Debian
sudo apt update && sudo apt upgrade -y

# For RHEL/CentOS
sudo dnf update -y
```

### 2. Install Basic Dependencies

```bash
# For Ubuntu/Debian
sudo apt install -y \
    build-essential \
    curl \
    wget \
    git \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# For RHEL/CentOS
sudo dnf install -y \
    gcc \
    gcc-c++ \
    make \
    curl \
    wget \
    git
```

## NVIDIA Driver Installation

### 1. Verify GPU Detection

```bash
# Check if GPU is detected by the system
lspci | grep -i nvidia
```

You should see output similar to:
```
01:00.0 VGA compatible controller: NVIDIA Corporation TU102 [NVIDIA RTX 6000/8000] (rev a1)
```

### 2. Install NVIDIA Drivers

#### Option A: Ubuntu (Recommended)

```bash
# Add NVIDIA package repositories
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update

# Install the recommended driver (525+ for RTX 6000)
sudo apt install -y nvidia-driver-535

# Reboot to load the driver
sudo reboot
```

#### Option B: RHEL/CentOS

```bash
# Install EPEL repository
sudo dnf install -y epel-release

# Install kernel headers and development packages
sudo dnf install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)

# Download and install NVIDIA driver
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.154.05/NVIDIA-Linux-x86_64-535.154.05.run
sudo bash NVIDIA-Linux-x86_64-535.154.05.run

# Reboot to load the driver
sudo reboot
```

### 3. Verify NVIDIA Driver Installation

```bash
# After reboot, verify driver installation
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.2    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 6000     Off  | 00000000:01:00.0 Off |                  Off |
| 30%   35C    P8    16W / 260W |      0MiB / 24576MiB |      0%      Default |
+-----------------------------------------------------------------------------+
```

## Docker and NVIDIA Container Toolkit Setup

### 1. Install Docker

#### Ubuntu/Debian

```bash
# Remove old versions
sudo apt remove -y docker docker-engine docker.io containerd runc

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to the docker group (logout/login required)
sudo usermod -aG docker $USER
```

#### RHEL/CentOS

```bash
# Add Docker repository
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# Install Docker
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to the docker group
sudo usermod -aG docker $USER
```

### 2. Install NVIDIA Container Toolkit

```bash
# Configure the repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the NVIDIA Container Toolkit packages
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. Verify Docker GPU Access

```bash
# Test GPU access from Docker
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

You should see the same `nvidia-smi` output as before, confirming Docker can access the GPU.

## Application Deployment

### 1. Clone the Repository

```bash
cd /opt
sudo git clone https://github.com/MRenAIAgent/agent_suite.git
cd agent_suite
sudo chown -R $USER:$USER .
```

### 2. Checkout the Deployment Branch

```bash
git fetch origin
git checkout claude/deploy-rtx6000-container-01SCA8NiR4UxCBSh1kShGzZe
```

### 3. Build the Application Container

```bash
cd /opt/agent_suite

# Build the GPU-enabled application container
docker build -f Dockerfile.gpu -t math-learning-gpu:latest .
```

### 4. Configure Environment Variables

Create a `.env` file with your configuration:

```bash
cat > math_learning/.env << 'EOF'
# API Keys (replace with your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Application Settings
LOG_LEVEL=INFO
EOF
```

### 5. Deploy with Docker Compose

```bash
cd math_learning

# Start all services with GPU support
docker compose -f docker-compose.gpu.yml up -d

# Check container status
docker compose -f docker-compose.gpu.yml ps

# View logs
docker compose -f docker-compose.gpu.yml logs -f
```

### 6. Verify Application is Running

```bash
# Check health of all containers
docker compose -f docker-compose.gpu.yml ps

# Test the API endpoint
curl http://localhost:8000/health

# Test GPU usage in the application
docker compose -f docker-compose.gpu.yml exec math-learning nvidia-smi
```

## GPU Verification

### 1. Check GPU Utilization

```bash
# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Or use the following for detailed monitoring
nvidia-smi dmon -s pucvmet
```

### 2. Verify CUDA is Available in the Application

```bash
# Execute a Python shell in the container
docker compose -f docker-compose.gpu.yml exec math-learning python3 << 'EOF'
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
EOF
```

Expected output:
```
CUDA Available: True
CUDA Device Count: 1
CUDA Device Name: NVIDIA RTX 6000 Ada Generation
CUDA Device Memory: 24.00 GB
```

### 3. Test Embedding Generation with GPU

```bash
docker compose -f docker-compose.gpu.yml exec math-learning python3 << 'EOF'
from sentence_transformers import SentenceTransformer
import torch

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Generate embeddings
sentences = ["This is a test sentence"] * 100
embeddings = model.encode(sentences, show_progress_bar=True)

print(f"Model device: {model.device}")
print(f"Embeddings shape: {embeddings.shape}")
EOF
```

## Troubleshooting

### GPU Not Detected in Container

```bash
# Check NVIDIA runtime is configured
docker info | grep -i runtime

# Ensure nvidia-container-runtime is listed
# If not, reconfigure:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### CUDA Out of Memory Errors

```bash
# Reduce batch size in environment variables
# Edit .env file and add/modify:
BATCH_SIZE=16
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Restart containers
docker compose -f docker-compose.gpu.yml restart
```

### Driver Version Mismatch

```bash
# Check driver and CUDA compatibility
nvidia-smi

# If CUDA version is incompatible, update the base image in Dockerfile.gpu
# Use a matching CUDA version, e.g., for driver 535.xx use CUDA 12.2
```

### Container Can't Access GPU

```bash
# Verify GPU is not locked by another process
nvidia-smi

# Check if GPU is in compute mode
nvidia-smi -q | grep "Compute Mode"

# If set to "Exclusive Process", change to default:
sudo nvidia-smi -c 0

# Restart the container
docker compose -f docker-compose.gpu.yml restart math-learning
```

### Performance Issues

```bash
# Enable GPU persistence mode for better performance
sudo nvidia-smi -pm 1

# Set GPU clock to maximum
sudo nvidia-smi -lgc 2100

# Monitor GPU utilization
nvidia-smi dmon -s pucvmet -c 100
```

### Network Issues

```bash
# Check if ports are accessible
sudo netstat -tlnp | grep -E '6333|6334|7474|7687|8000'

# If firewall is enabled, allow required ports
sudo ufw allow 6333/tcp  # Qdrant HTTP
sudo ufw allow 6334/tcp  # Qdrant gRPC
sudo ufw allow 7474/tcp  # Neo4j HTTP
sudo ufw allow 7687/tcp  # Neo4j Bolt
sudo ufw allow 8000/tcp  # Application API
```

### Container Logs

```bash
# View logs for specific service
docker compose -f docker-compose.gpu.yml logs math-learning

# Follow logs in real-time
docker compose -f docker-compose.gpu.yml logs -f math-learning

# View last 100 lines
docker compose -f docker-compose.gpu.yml logs --tail=100 math-learning
```

## Maintenance

### Updating the Application

```bash
cd /opt/agent_suite
git pull origin claude/deploy-rtx6000-container-01SCA8NiR4UxCBSh1kShGzZe
docker compose -f math_learning/docker-compose.gpu.yml down
docker build -f Dockerfile.gpu -t math-learning-gpu:latest .
docker compose -f math_learning/docker-compose.gpu.yml up -d
```

### Backup Database Volumes

```bash
# Backup Qdrant data
docker run --rm -v math_learning_qdrant_storage:/data -v $(pwd):/backup ubuntu tar czf /backup/qdrant_backup_$(date +%Y%m%d).tar.gz /data

# Backup Neo4j data
docker run --rm -v math_learning_neo4j_data:/data -v $(pwd):/backup ubuntu tar czf /backup/neo4j_backup_$(date +%Y%m%d).tar.gz /data
```

### Monitoring GPU Health

```bash
# Create a monitoring script
cat > /usr/local/bin/gpu-monitor.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== GPU Status ==="
    nvidia-smi
    echo ""
    echo "=== Container GPU Usage ==="
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
    sleep 5
done
EOF

chmod +x /usr/local/bin/gpu-monitor.sh

# Run the monitor
/usr/local/bin/gpu-monitor.sh
```

## Security Considerations

1. **API Keys**: Never commit API keys to the repository. Use environment variables or secrets management.

2. **Network Security**:
   ```bash
   # Only expose necessary ports
   # Use a reverse proxy (nginx) for HTTPS
   # Consider using Docker networks for internal communication
   ```

3. **User Permissions**:
   ```bash
   # Run containers as non-root user
   # Add to docker-compose.yml:
   # user: "1000:1000"
   ```

4. **Regular Updates**:
   ```bash
   # Keep system updated
   sudo apt update && sudo apt upgrade -y

   # Update Docker images regularly
   docker compose -f math_learning/docker-compose.gpu.yml pull
   ```

## Performance Tuning

### GPU Memory Optimization

Edit `.env`:
```bash
# For 24GB RTX 6000, allocate memory wisely
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
```

### Database Tuning

Neo4j configuration (`docker-compose.gpu.yml`):
```yaml
NEO4J_dbms_memory_heap_max__size: 8G
NEO4J_dbms_memory_pagecache_size: 4G
```

### Application Tuning

```bash
# Increase worker processes for API
WORKERS=4
THREADS=2
```

## Additional Resources

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [RTX 6000 Specifications](https://www.nvidia.com/en-us/design-visualization/rtx-6000/)
- [PyTorch CUDA Best Practices](https://pytorch.org/docs/stable/notes/cuda.html)

## Support

For issues related to:
- Application bugs: Open an issue on GitHub
- GPU/Driver issues: Check NVIDIA forums
- Docker issues: Check Docker documentation
