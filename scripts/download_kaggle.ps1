$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$rawDir = Join-Path $projectRoot "data\raw"
New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    throw "Kaggle CLIが見つかりません。先に 'pip install kaggle' を実行してください。"
}

$slugs = @(
    "zaldyjr/cacao-diseases-in-davao",
    "zaldyjr/cacao-diseases"
)

$downloaded = $false
foreach ($slug in $slugs) {
    try {
        Write-Host "Kaggle datasetを取得: $slug"
        kaggle datasets download -d $slug -p $rawDir --unzip
        if ($LASTEXITCODE -eq 0) {
            $downloaded = $true
            break
        }
    } catch {
        Write-Warning "取得に失敗: $slug"
    }
}

if (-not $downloaded) {
    throw "Kaggle datasetを取得できませんでした。kaggle.jsonとdataset slugを確認してください。"
}

Write-Host "取得完了: $rawDir"

