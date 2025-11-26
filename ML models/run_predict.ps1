param(
  [string[]] $Ignore = @("7","12"),
  [string]   $Python = "python",
  [int[]]    $PCA    = @(0,5)  # 0 = sans PCA, 5 = avec PCA(5)
)

if (!(Test-Path reports)) { New-Item -ItemType Directory reports | Out-Null }

$models = @("xgb","lgbm")

foreach ($m in $models) {
  foreach ($p in $PCA) {
    $name = if ($p -eq 0) { "base" } else { "pca$p" }
    $fig  = "reports\roc_${m}_$name.png"
    $out  = "reports\report_${m}_$name.txt"

    $args = @("--model", $m, "--ignore") + $Ignore + @("--save-fig", $fig)
    if ($p -gt 0) { $args += @("--pca-components", "$p") }

    & $Python "predict.py" @args | Tee-Object -FilePath $out
  }
}

Write-Host "Terminé. Graphes et reports dans 'reports\'."
