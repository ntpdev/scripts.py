function z {
    $br = Invoke-Expression "git branch --show-current"
    if ( ($br.Length -gt 0) -and ($br -ne "masterx")) {
        Write-Output "git push --set-upstream origin $br"
    } else {
        Write-Output "cannot create tracking branch"
    }
}

# get list of files changed
$f = Invoke-Expression "git diff-index --name-only HEAD"
if ($f.Length -gt 0) {
    Invoke-Expression "7z a tmp $f"
}