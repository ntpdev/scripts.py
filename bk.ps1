# backup directories to zip
$zip = get-command 7z.exe | select-object -ExpandProperty source
$outpath="c:\temp"
dir -Directory | 
% {
    $d = $_.Name
    $dn = $d -replace "\.","-"
    $a = @('a', "$outpath\$dn", "$d\","-xr!.gradle", "-xr!.git", "-xr!.idea", "-xr!build", "-xr!__pycache__")
    Write-Output "Backing up $d"
    &$zip $a
  }