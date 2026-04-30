param(
    [string]$DatasetPath = "data",
    [string]$ColmapPath = "",
    [switch]$Clean,
    [switch]$SparseOnly,
    [switch]$UseGpu
)

$ErrorActionPreference = "Stop"

function Resolve-ColmapCommand {
    param([string]$InputPath)

    if ($InputPath -ne "") {
        if (Test-Path $InputPath -PathType Leaf) {
            return (Resolve-Path $InputPath).Path
        }
        $batPath = Join-Path $InputPath "COLMAP.bat"
        $exePath = Join-Path $InputPath "bin\colmap.exe"
        if (Test-Path $batPath -PathType Leaf) {
            return (Resolve-Path $batPath).Path
        }
        if (Test-Path $exePath -PathType Leaf) {
            return (Resolve-Path $exePath).Path
        }
        throw "Cannot find COLMAP.bat or bin\colmap.exe under: $InputPath"
    }

    $cmd = Get-Command colmap -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $cmd = Get-Command COLMAP.bat -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $defaultPath = "C:\Users\Administrator\Downloads\colmap-x64-windows-cuda"
    $defaultBat = Join-Path $defaultPath "COLMAP.bat"
    if (Test-Path $defaultBat -PathType Leaf) {
        return (Resolve-Path $defaultBat).Path
    }

    throw "COLMAP is not available in PATH. Add COLMAP to PATH or pass -ColmapPath `"C:\Users\Administrator\Downloads\colmap-x64-windows-cuda`"."
}

function Invoke-Colmap {
    param(
        [string]$CommandPath,
        [string[]]$Arguments
    )

    & $CommandPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "COLMAP command failed with exit code $LASTEXITCODE`: $CommandPath $($Arguments -join ' ')"
    }
}

$ImagePath = Join-Path $DatasetPath "images"
$ColmapCommand = Resolve-ColmapCommand $ColmapPath
$OutputPath = Join-Path $DatasetPath "colmap"
$SparsePath = Join-Path $OutputPath "sparse"
$DensePath = Join-Path $OutputPath "dense"
$DatabasePath = Join-Path $OutputPath "database.db"

if ($Clean -and (Test-Path $OutputPath)) {
    $resolvedDataset = (Resolve-Path $DatasetPath).Path
    $resolvedOutput = (Resolve-Path $OutputPath).Path
    if (-not $resolvedOutput.StartsWith($resolvedDataset)) {
        throw "Refuse to clean path outside dataset directory: $resolvedOutput"
    }
    Write-Host "Cleaning previous COLMAP output: $resolvedOutput"
    Remove-Item -LiteralPath $resolvedOutput -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $SparsePath | Out-Null
New-Item -ItemType Directory -Force -Path $DensePath | Out-Null

$gpuFlag = if ($UseGpu) { "1" } else { "0" }

Write-Host "Using COLMAP: $ColmapCommand"
Write-Host "=== Step 1: Feature Extraction ==="
Invoke-Colmap $ColmapCommand @(
    "feature_extractor",
    "--database_path", $DatabasePath,
    "--image_path", $ImagePath,
    "--ImageReader.camera_model", "PINHOLE",
    "--ImageReader.single_camera", "1",
    "--FeatureExtraction.use_gpu", $gpuFlag
)

Write-Host "=== Step 2: Feature Matching ==="
Invoke-Colmap $ColmapCommand @(
    "exhaustive_matcher",
    "--database_path", $DatabasePath,
    "--FeatureMatching.use_gpu", $gpuFlag
)

Write-Host "=== Step 3: Sparse Reconstruction (Bundle Adjustment) ==="
Invoke-Colmap $ColmapCommand @(
    "mapper",
    "--database_path", $DatabasePath,
    "--image_path", $ImagePath,
    "--output_path", $SparsePath
)

if ($SparseOnly) {
    Write-Host "=== Sparse-only mode: skip dense reconstruction ==="
    Write-Host "Sparse result: $SparsePath\0"
    exit 0
}

Write-Host "=== Step 4: Image Undistortion ==="
Invoke-Colmap $ColmapCommand @(
    "image_undistorter",
    "--image_path", $ImagePath,
    "--input_path", (Join-Path $SparsePath "0"),
    "--output_path", $DensePath
)

Write-Host "=== Step 5: Dense Reconstruction (Patch Match Stereo) ==="
Invoke-Colmap $ColmapCommand @(
    "patch_match_stereo",
    "--workspace_path", $DensePath,
    "--PatchMatchStereo.gpu_index", $(if ($UseGpu) { "0" } else { "-1" })
)

Write-Host "=== Step 6: Stereo Fusion ==="
Invoke-Colmap $ColmapCommand @(
    "stereo_fusion",
    "--workspace_path", $DensePath,
    "--output_path", (Join-Path $DensePath "fused.ply")
)

Write-Host "=== Done! ==="
Write-Host "Sparse: $SparsePath\0"
Write-Host "Dense:  $DensePath\fused.ply"
