# =============================================================
# GLDT Streamlit Dashboard Startup Script
# =============================================================
# Usage: .\scripts\start_streamlit.ps1
#        .\scripts\start_streamlit.ps1 -BackendUrl http://localhost:8000
# =============================================================

param(
    [int]   $Port       = 8501,
    [string]$BackendUrl = "http://localhost:8000"
)

$env:BACKEND_URL = $BackendUrl

Write-Host ""
Write-Host "=== GLDT Streamlit Dashboard ===" -ForegroundColor Cyan
Write-Host "  Backend : $BackendUrl"
Write-Host "  Port    : $Port"
Write-Host "  Pages   :"
Write-Host "    01 Overview (global heatmap)"
Write-Host "    02 Country Deep-Dive"
Write-Host "    03 Data Explorer"
Write-Host "    04 Escalation Forecast  [Phase 3]"
Write-Host "    05 GNN Network          [Phase 3]"
Write-Host "    06 RAG Advisory         [Phase 3]"
Write-Host ""

streamlit run streamlit_app\Home.py `
    --server.port $Port `
    --server.address localhost `
    --server.headless true
