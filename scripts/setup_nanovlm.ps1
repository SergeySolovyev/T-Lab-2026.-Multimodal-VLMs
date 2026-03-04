param(
  [string]$Target = "external/nanoVLM"
)

$ErrorActionPreference = "Stop"

if (Test-Path $Target) {
  Write-Host "NanoVLM already exists at $Target"
  exit 0
}

git clone https://github.com/huggingface/nanoVLM.git $Target
Write-Host "Cloned nanoVLM to $Target"
