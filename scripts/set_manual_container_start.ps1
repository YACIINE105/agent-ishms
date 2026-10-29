param(
    [string]$DockerContext = 'desktop-linux',
    [string[]]$ComposePaths = @(),
    [switch]$ComposeOnly
)

$ErrorActionPreference = 'Stop'

# Resolve and validate every Compose target before changing any settings.
if ($ComposePaths.Count -eq 0) {
    $scriptDirectory = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $PSScriptRoot).ProviderPath)
    $workspaceRoot = [System.IO.Directory]::GetParent($scriptDirectory).FullName
    $projectsRoot = [System.IO.Directory]::GetParent($workspaceRoot).FullName
    $ComposePaths = @(
        ([System.IO.Path]::Combine($workspaceRoot, 'docker\docker-compose.yml')),
        ([System.IO.Path]::Combine($projectsRoot, 'rag-ishms\docker\docker-compose.yml')),
        ([System.IO.Path]::Combine($projectsRoot, 'rag-ishms\docker\release\compose.yml'))
    )
}

$resolvedComposePaths = @()
foreach ($composePath in $ComposePaths) {
    if (-not (Test-Path -LiteralPath $composePath -PathType Leaf)) {
        throw "Compose file not found: $composePath. No settings were changed by this run."
    }
    $resolvedComposePath = Resolve-Path -LiteralPath $composePath
    if ($resolvedComposePath.Provider.Name -ne 'FileSystem') {
        throw "Compose path must refer to a filesystem file: $composePath"
    }
    $fileSystemPath = [System.IO.Path]::GetFullPath($resolvedComposePath.ProviderPath)
    $resolvedComposePaths += $fileSystemPath
    Write-Host "Compose target: $fileSystemPath"
}

if (-not $ComposeOnly) {
    # Step 1 : verify that the selected context is the local Linux Docker engine.
    $dockerCommand = Get-Command docker.exe -ErrorAction Stop
    $dockerPath = $dockerCommand.Source
    $dockerEndpoint = & $dockerPath context inspect $DockerContext --format '{{.Endpoints.docker.Host}}'
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not inspect the Docker context. Check docker context ls first.'
    }
    if ($dockerEndpoint -notlike 'npipe://*') {
        throw 'This script only changes a local Windows named-pipe Docker context.'
    }

    $engineType = & $dockerPath --context $DockerContext info --format '{{.OSType}}'
    if ($LASTEXITCODE -ne 0 -or $engineType -ne 'linux') {
        throw 'The selected Linux engine must be running before changing restart policies.'
    }

    # Step 2 : change existing containers without stopping or recreating them.
    $containerIds = @(& $dockerPath --context $DockerContext ps -aq)
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not list the existing containers.'
    }

    foreach ($containerId in $containerIds) {
        & $dockerPath --context $DockerContext update --restart=no $containerId
        if ($LASTEXITCODE -ne 0) {
            throw "Could not update container $containerId. Some earlier updates may have succeeded."
        }
    }

    foreach ($containerId in $containerIds) {
        $policy = & $dockerPath --context $DockerContext inspect --format '{{.HostConfig.RestartPolicy.Name}}' $containerId
        if ($LASTEXITCODE -ne 0 -or $policy -ne 'no') {
            throw "Restart policy verification failed for container $containerId."
        }
    }

    Write-Host "Verified restart=no for $($containerIds.Count) existing containers in $DockerContext."
} else {
    Write-Host 'Compose-only mode: using the previously verified container policies.'
}

# Step 3 : preserve the choice in the known Compose source files.
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
$restartPattern = '(?m)^([ \t]*restart:[ \t]*)([^#\r\n]+?)([ \t]*(?:#[^\r\n]*)?)(\r?)$'

foreach ($fileSystemPath in $resolvedComposePaths) {
    $original = [System.IO.File]::ReadAllText($fileSystemPath)
    $updated = [regex]::Replace($original, $restartPattern, '${1}"no"${3}${4}')
    if ($updated -ne $original) {
        Copy-Item -LiteralPath $fileSystemPath -Destination "$fileSystemPath.before-manual-start-$timestamp"
        [System.IO.File]::WriteAllText($fileSystemPath, $updated, $utf8WithoutBom)
    }
    if ([System.IO.File]::ReadAllText($fileSystemPath) -ne $updated) {
        throw "Compose file verification failed: $fileSystemPath"
    }
    Write-Host "Manual-start Compose settings: $fileSystemPath"
}

Write-Host 'Verified manual-start settings in all selected Compose files.'
Write-Host 'Containers and data were preserved. Currently running containers were left running.'
