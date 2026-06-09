$ErrorActionPreference = "Stop"

$PaperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $PaperDir
$OutName = "when_plausible_videos_lie_iclr_submission.pdf"
$LocalOut = Join-Path $PaperDir $OutName
$DownloadsOut = Join-Path (Join-Path $env:USERPROFILE "Downloads") $OutName

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

    Copy-Item -LiteralPath "main.pdf" -Destination $LocalOut -Force
    Copy-Item -LiteralPath "main.pdf" -Destination $DownloadsOut -Force
    Write-Host "Built $LocalOut"
    Write-Host "Copied $DownloadsOut"
} finally {
    Pop-Location
}
