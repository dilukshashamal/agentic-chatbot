---
name: docker-optimization
description: "Optimize Docker Compose setups to reduce memory, CPU, startup time, and disk usage. Use when: docker containers consume excessive resources, startup is slow, builds take too long, or disk space is being wasted by volumes and images."
argument-hint: "Optional project type (e.g., microservices, monolith, data pipeline)"
---

# Docker Optimization

## When to Use

- Containers are consuming excessive memory during `docker-compose up`
- Startup time is slow or services fail to start
- CPU usage is consistently high
- Disk space is being wasted by volumes, images, or build layers
- Build times are lengthy
- Development workflow has unnecessary friction (hot reload issues, dependency conflicts)

## Overview

Docker Compose stacks can bloat quickly when multiple services run simultaneously. This skill provides a systematic approach to identify bottlenecks and apply targeted optimizations across four areas: **resource limits**, **image optimization**, **volume management**, and **development workflow**.

---

## 1. Diagnose Current Resource Usage

### Check Real-Time Resource Consumption

```bash
# Display live memory, CPU, and disk I/O for all containers
docker stats --no-stream

# View detailed resource limits and current usage
docker inspect <container-name> | grep -A 10 "HostConfig"

# List image sizes
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# List volume sizes
docker volume ls --format "table {{.Name}}\t{{.Driver}}" | xargs -I {} docker volume inspect {}
```

### Check Compose Config Issues

```bash
# Validate your docker-compose.yml
docker-compose config --services

# Display extended compose information
docker-compose ps
```

---

## 2. Optimize Memory Usage

### Set Memory Limits

Memory limits prevent containers from consuming unbounded system resources. Add to `docker-compose.yml`:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 1G # Hard limit
        reservations:
          memory: 512M # Soft limit (guaranteed)

  frontend:
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  db:
    deploy:
      resources:
        limits:
          memory: 2G # Databases need more RAM
        reservations:
          memory: 1G

  mlflow:
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M

  prometheus:
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  grafana:
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M
```

**Memory allocation strategy:**

- **Database (PostgreSQL)**: 1-2 GB (query caching, buffers)
- **Backend API (Python/FastAPI)**: 512 MB - 1 GB (frameworks + dependencies)
- **Frontend (Next.js)**: 512 MB (next server + build cache)
- **Monitoring (Prometheus/Grafana)**: 256-512 MB each
- **MLflow**: 512 MB - 1 GB (artifact storage, model loading)

### Reduce Memory Pressure

**For Python applications (backend):**

```bash
# Add these environment variables to reduce memory footprint
PYTHONUNBUFFERED=1           # Use unbuffered stdout/stderr
PYTHONOPTIMIZE=2             # Optimize bytecode (-OO)
MALLOC_TRIM_THRESHOLD_=128000 # Reduce memory fragmentation
```

**For Node.js (frontend):**

```bash
# Limit Node.js heap size
NODE_OPTIONS="--max-old-space-size=512"
```

---

## 3. Optimize Build Times

### Use Multi-Stage Builds

Keep only production artifacts in final images. Example structure:

```dockerfile
# Stage 1: Build
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
```

Benefits: Smaller final image (70-80% reduction), faster deploys, less disk usage.

### Leverage Build Cache

```bash
# Build with no cache (slow, use sparingly)
docker-compose build --no-cache

# Build with cache (default, faster)
docker-compose build

# Use BuildKit for better caching
DOCKER_BUILDKIT=1 docker-compose build
```

**Order Dockerfile commands strategically:**

1. Base image + system dependencies (rarely changes)
2. Language/framework setup (medium frequency)
3. Application dependencies (frequent changes)
4. Application code (most frequent changes)

---

## 4. Optimize Disk Usage

### Clean Up Unused Resources

```bash
# Remove dangling images (unused layers)
docker image prune -a --force

# Remove stopped containers
docker container prune --force

# Remove unused volumes (⚠️ CAREFUL: deletes data)
docker volume prune --force

# Remove unused networks
docker network prune --force

# One-command cleanup
docker system prune -a --volumes
```

### Reduce Volume Size

```bash
# Find largest volumes
docker volume ls -q | xargs docker volume inspect | grep -A 2 "Name\|Mountpoint"

# Inspect volume usage
du -sh /var/lib/docker/volumes/<volume-name>/_data

# Archive old data periodically
tar -czf backup.tar.gz /var/lib/docker/volumes/<volume-name>/_data
```

### Optimize Database Volumes

Add to PostgreSQL service:

```yaml
db:
  environment:
    # Reduce shared buffers if memory is constrained
    POSTGRES_INITDB_ARGS: "-c shared_buffers=128MB -c effective_cache_size=256MB"
  command:
    - "postgres"
    - "-c"
    - "shared_buffers=128MB"
    - "-c"
    - "work_mem=8MB"
```

---

## 5. Optimize Development Workflow

### Disable Hot Reload Where Unnecessary

Hot reload (file watching) adds CPU overhead. Disable if not actively developing:

```yaml
services:
  backend:
    # Remove --reload flag or set to false
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000
    # Only use reload when actively developing

  frontend:
    # Disable Chokidar polling if on native Docker (WSL/Mac native)
    environment:
      CHOKIDAR_USEPOLLING: "false" # Set to true only on Windows + VirtualBox
      WATCHPACK_POLLING: "false" # Set to true only on Windows + VirtualBox
```

### Use Named Volumes for Better Performance

Named volumes are faster than bind mounts on Windows/Mac:

```yaml
volumes:
  node_modules_cache:
  venv_cache:

services:
  frontend:
    volumes:
      - ./frontend:/app/frontend
      - node_modules_cache:/app/frontend/node_modules # Cached, not bind-mounted

  backend:
    volumes:
      - ./backend:/app/backend
      - venv_cache:/app/venv # Faster than mounting venv
```

### Selective Bind Mounts

Only bind-mount what you're actively editing:

```yaml
backend:
  volumes:
    # Only mount source code, not .venv or __pycache__
    - ./backend/app:/app/backend/app
    - ./backend/migrations:/app/backend/migrations

frontend:
  volumes:
    # Only mount source, not node_modules or .next
    - ./frontend/app:/app/frontend/app
    - ./frontend/components:/app/frontend/components
```

---

## 6. Docker Desktop / Host System Optimization

### Windows (WSL2)

```powershell
# Check WSL2 resource limits (~WSLConfig file)
cat $env:USERPROFILE\.wslconfig
```

Add to `~/.wslconfig`:

```ini
[wsl2]
memory=4GB          # Allocate 4GB to WSL2
processors=4        # Use 4 CPU cores
swap=2GB            # Swap space
localhostForwarding=true
```

Restart WSL: `wsl --shutdown`

### macOS

```bash
# Check Docker Desktop memory allocation
docker run --rm --privileged alpine sysctl hw.memsize

# Edit via Docker Desktop GUI:
# Docker > Preferences > Resources > Memory (default 2GB, increase to 4GB+)
# CPU Shares (default 2, increase if needed)
# Swap (default 1GB)
```

### Linux

```bash
# Check available system memory
free -h

# No configuration needed; Docker uses host resources directly
# Optionally limit globally in /etc/docker/daemon.json:
{
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

---

## 7. Production-Ready Checklist

- [ ] Memory and CPU limits defined for all services
- [ ] Multi-stage builds used; final images <500MB each
- [ ] Unused images, volumes, and containers removed
- [ ] Health checks configured for critical services
- [ ] Restart policies set (`unless-stopped` or `on-failure`)
- [ ] Secrets passed via `.env` files (not hardcoded)
- [ ] Logging configured (log rotation, size limits)
- [ ] Read-only root filesystem where possible
- [ ] Non-root users running containers
- [ ] Regular backups of persistent volumes

---

## Quick Reference Commands

```bash
# Full diagnostic
docker stats
docker-compose config
docker images
docker volume ls

# Cleanup
docker system prune -a --volumes

# Rebuild with optimizations
docker-compose down
docker-compose build --no-cache
docker-compose up --no-deps

# Monitor during startup
docker-compose up --scale backend=1
docker stats --no-stream
```

---

## References & Next Steps

**Optimize for your specific stack:**

- Python/FastAPI: Reduce dependencies, use slim base images, pin versions
- Next.js: Disable polling on native Docker, use named volumes for node_modules
- PostgreSQL: Tune `shared_buffers`, `work_mem`, `effective_cache_size`
- Monitoring: Consider running only in production; disable during dev if unused

**Further reading:**

- Docker memory limits: https://docs.docker.com/config/containers/resource_constraints/
- Compose best practices: https://docs.docker.com/compose/production/
- Python image optimization: https://docs.docker.com/language/python/optimize-build/
