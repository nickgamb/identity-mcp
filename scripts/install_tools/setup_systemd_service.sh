#!/bin/bash
# Script to set up systemd service for Docker Compose stack

set -e

echo "=== Setting up systemd service for Docker Compose stack ==="
echo ""

SERVICE_FILE="/etc/systemd/system/identity-mcp.service"
WORK_DIR="/home/nick/ai"

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "This script must be run with sudo"
    echo "Usage: sudo bash setup_systemd_service.sh"
    exit 1
fi

# Create the service file
echo "Creating systemd service file..."
cat > "$SERVICE_FILE" << 'EOF'
[Unit]
Description=Identity MCP Stack
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/nick/ai
ExecStart=/usr/bin/docker compose --profile hf --profile identity up -d
ExecStop=/usr/bin/docker compose --profile hf --profile identity down
TimeoutStartSec=0
User=nick
Group=nick

[Install]
WantedBy=multi-user.target
EOF

# Set correct permissions
chmod 644 "$SERVICE_FILE"
echo "✓ Service file created at $SERVICE_FILE"

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload
echo "✓ Systemd daemon reloaded"

# Enable the service
echo "Enabling service to start on boot..."
systemctl enable identity-mcp.service
echo "✓ Service enabled"

# Check if user wants to start it now
echo ""
read -p "Start the service now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting service..."
    systemctl start identity-mcp.service
    echo "✓ Service started"
    
    echo ""
    echo "Service status:"
    systemctl status identity-mcp.service --no-pager
else
    echo "Service will start automatically on next boot"
fi

echo ""
echo "=== Setup complete ==="
echo "Service: identity-mcp.service"
echo "Status: systemctl status identity-mcp.service"
echo "Start: systemctl start identity-mcp.service"
echo "Stop: systemctl stop identity-mcp.service"
echo "Disable: systemctl disable identity-mcp.service"

