param(
    [string]$DesktopCopy = ""
)

$ErrorActionPreference = "Stop"

$PaperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $PaperDir
$FinalDir = Join-Path $PaperDir "final"
$FinalOut = Join-Path $FinalDir "best of n video transformer world model-v4.pdf"
New-Item -ItemType Directory -Force $FinalDir | Out-Null

Push-Location $PaperDir
try {
    $latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
    $perl = Get-Command perl -ErrorAction SilentlyContinue
    $builtWithLatexmk = $false
    if ($latexmk -and $perl) {
        & latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
        $builtWithLatexmk = (($LASTEXITCODE -eq 0) -and (Test-Path "main.pdf"))
        if (-not $builtWithLatexmk) {
            Write-Warning "latexmk failed or did not produce main.pdf; falling back to pdflatex/bibtex."
        }
    } elseif ($latexmk -and -not $perl) {
        Write-Warning "latexmk is installed but Perl is missing; using pdflatex/bibtex fallback."
    }

    if (-not $builtWithLatexmk) {
        & pdflatex -interaction=nonstopmode -halt-on-error main.tex
        & bibtex main
        & pdflatex -interaction=nonstopmode -halt-on-error main.tex
        & pdflatex -interaction=nonstopmode -halt-on-error main.tex
    }

    if (-not (Test-Path "main.pdf")) {
        throw "LaTeX build did not produce main.pdf"
    }

    Copy-Item -LiteralPath "main.pdf" -Destination $FinalOut -Force
    if ($DesktopCopy) {
        $DesktopDir = Split-Path -Parent $DesktopCopy
        if ($DesktopDir) {
            New-Item -ItemType Directory -Force -Path $DesktopDir | Out-Null
        }
        Copy-Item -LiteralPath $FinalOut -Destination $DesktopCopy -Force
    }
    Remove-Item -Force "main.pdf" -ErrorAction SilentlyContinue
    Write-Host "Built $FinalOut"
    if ($DesktopCopy) {
        Write-Host "Built $DesktopCopy"
    }
} finally {
    Pop-Location
}
