# Directory structure update script
$projectRoot = "."
$outputFile = "./.notes/directory_structure.md"

# Directories to exclude
$excludeDirs = @(
    '.git',
    'node_modules',
    'dist',
    'build',
    'coverage',
    '.cache',
    '.npm',
    '.yarn',
    'tmp',
    'temp',
    'logs',
    'vpn'
)

# Files to exclude
$excludeFiles = @(
    '*.log',
    '*.min.js',
    '*.min.css',
    '*.bak',
    '*.backup',
    '*~',
    '.DS_Store',
    'Thumbs.db',
    'package-lock.json',
    'yarn.lock'
)

# Generate directory listing
function Get-FormattedDirectory {
    param (
        [string]$Path,
        [int]$Indent = 0
    )

    $indentString = "    " * $Indent
    $content = ""

    try {
        $items = Get-ChildItem -Path $Path -Force -ErrorAction SilentlyContinue | 
                Where-Object { 
                    $shouldInclude = $true
                    if ($_.PSIsContainer) {
                        $shouldInclude = $excludeDirs -notcontains $_.Name
                    } else {
                        $shouldInclude = -not ($excludeFiles | Where-Object { $_.Name -like $_ })
                    }
                    $shouldInclude
                }

        foreach ($item in $items) {
            if ($item.PSIsContainer) {
                $content += "$indentString- **$($item.Name)/**`n"
                $content += Get-FormattedDirectory -Path $item.FullName -Indent ($Indent + 1)
            } else {
                $content += "$indentString- $($item.Name)`n"
            }
        }
    } catch {
        Write-Warning -Message "Error accessing path: $Path"
    }
    
    return $content
}

# Generate content for markdown file
$markdownContent = @"
# Current Directory Structure

## Core Components
Main project directories and files, excluding build artifacts, dependencies, and temporary files.

\`\`\`
$(Get-FormattedDirectory -Path $projectRoot)
\`\`\`

Note: The following are excluded for clarity:
- Build directories (dist/, build/)
- Dependencies (node_modules/)
- Version control (.git/)
- Temporary and log files
- Cache directories
- VPN directories (vpn/)
"@

# Output to file
$null = New-Item -Path (Split-Path -Path $outputFile) -ItemType Directory -Force -ErrorAction SilentlyContinue
Set-Content -Path $outputFile -Value $markdownContent -Encoding UTF8

Write-Output "Directory structure updated in $outputFile" 