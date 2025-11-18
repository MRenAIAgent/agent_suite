# Quick Start Guide: RTX 6000 Deployment

This is a condensed guide for deploying the Math Learning application with RTX 6000 GPU support. For detailed instructions, see [RTX6000_DEPLOYMENT_GUIDE.md](docs/RTX6000_DEPLOYMENT_GUIDE.md).

## Prerequisites

- Ubuntu 20.04/22.04 or RHEL 8/9 VM
- NVIDIA RTX 6000 GPU (or compatible)
- Minimum 16GB RAM, 50GB disk space
- Root or sudo access

## Quick Installation Steps

### 1. Install NVIDIA Drivers

```bash
# Ubuntu
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot
```

Verify installation:
```bash
nvidia-smi
```

### 2. Install Docker

```bash
# Ubuntu
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

Logout and login again for group changes to take effect.

### 3. Install NVIDIA Container Toolkit

```bash
# Configure repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### 4. Clone and Deploy Application

```bash
# Clone repository
git clone https://github.com/MRenAIAgent/agent_suite.git
cd agent_suite
git checkout claude/deploy-rtx6000-container-01SCA8NiR4UxCBSh1kShGzZe

# Configure environment
cd math_learning
cp .env.example .env
nano .env  # Update API keys

# Run deployment script
cd ..
./scripts/deploy_rtx6000.sh check    # Check prerequisites
./scripts/deploy_rtx6000.sh deploy   # Deploy application
```

### 5. Verify Deployment

```bash
# Check service status
./scripts/deploy_rtx6000.sh status

# Test GPU in container
./scripts/deploy_rtx6000.sh gpu-test

# View logs
./scripts/deploy_rtx6000.sh logs
```

## Access the Application

- **API**: http://your-vm-ip:8000
- **Qdrant**: http://your-vm-ip:6333
- **Neo4j**: http://your-vm-ip:7474 (user: neo4j, password: password)

## Common Commands

```bash
# Start services
./scripts/deploy_rtx6000.sh start

# Stop services
./scripts/deploy_rtx6000.sh stop

# Restart services
./scripts/deploy_rtx6000.sh restart

# View status
./scripts/deploy_rtx6000.sh status

# View logs
./scripts/deploy_rtx6000.sh logs

# Test GPU
./scripts/deploy_rtx6000.sh gpu-test

# Complete cleanup (removes all data!)
./scripts/deploy_rtx6000.sh cleanup
```

## Manual Deployment (Alternative)

If you prefer manual deployment without the script:

```bash
# Build image
docker build -f Dockerfile.gpu -t math-learning-gpu:latest .

# Start services
cd math_learning
docker compose -f docker-compose.gpu.yml up -d

# Check status
docker compose -f docker-compose.gpu.yml ps

# View logs
docker compose -f docker-compose.gpu.yml logs -f
```

## Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check Qdrant
curl http://localhost:6333/health

# Check Neo4j
curl http://localhost:7474

# Check GPU usage
nvidia-smi
watch -n 1 nvidia-smi  # Monitor in real-time
```

## Troubleshooting Quick Fixes

### GPU not detected in container

```bash
# Reconfigure NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Restart containers
./scripts/deploy_rtx6000.sh restart
```

### Out of memory errors

```bash
# Edit .env file
nano math_learning/.env

# Add/modify:
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Restart
./scripts/deploy_rtx6000.sh restart
```

### Port conflicts

```bash
# Check what's using ports
sudo netstat -tlnp | grep -E '6333|7474|7687|8000'

# Kill conflicting processes or change ports in docker-compose.gpu.yml
```

### Containers won't start

```bash
# Check logs
docker compose -f math_learning/docker-compose.gpu.yml logs

# Check disk space
df -h

# Check memory
free -h

# Restart Docker daemon
sudo systemctl restart docker
```

## Performance Optimization

```bash
# Enable GPU persistence mode (better performance)
sudo nvidia-smi -pm 1

# Set maximum GPU clock
sudo nvidia-smi -lgc 2100

# Monitor GPU utilization
nvidia-smi dmon -s pucvmet
```

## Updating the Application

```bash
# Pull latest changes
cd agent_suite
git pull origin claude/deploy-rtx6000-container-01SCA8NiR4UxCBSh1kShGzZe

# Rebuild and redeploy
./scripts/deploy_rtx6000.sh stop
./scripts/deploy_rtx6000.sh build
./scripts/deploy_rtx6000.sh start
```

## Backup Data

```bash
# Backup Qdrant
docker run --rm -v math_learning_qdrant_storage:/data -v $(pwd):/backup ubuntu tar czf /backup/qdrant_backup.tar.gz /data

# Backup Neo4j
docker run --rm -v math_learning_neo4j_data:/data -v $(pwd):/backup ubuntu tar czf /backup/neo4j_backup.tar.gz /data
```

## Security Checklist

- [ ] Update API keys in `.env` file
- [ ] Change default Neo4j password
- [ ] Configure firewall (ufw/firewalld)
- [ ] Enable HTTPS with reverse proxy
- [ ] Set up log rotation
- [ ] Configure automated backups
- [ ] Enable GPU persistence mode
- [ ] Set up monitoring (Prometheus/Grafana)

## Support

For detailed documentation, see:
- [Full Deployment Guide](docs/RTX6000_DEPLOYMENT_GUIDE.md)
- [Math Learning README](math_learning/README.md)
- [GitHub Issues](https://github.com/MRenAIAgent/agent_suite/issues)

## Next Steps

1. Configure your API keys in `math_learning/.env`
2. Test the API endpoints
3. Set up HTTPS with nginx
4. Configure automated backups
5. Set up monitoring and alerting
6. Review security settings
7. Configure log rotation

Happy deploying! 🚀
