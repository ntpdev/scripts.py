# use to import this function into the current session
# Import-Module -Name .\examples1.ps1

function add ($x, $y) { $x + $y }

function sub { param($x, $y) $x - $y }

function div { param($x, $y=1) $x / $y }

# filter can be used to write fuctions that work effeciently with a pipeline 
filter mult { $_ * 2 }
1,2,3 | mult

# write text stream to file
1..15 | 
    % {"a{0:d2}b" -f $_} |
    Set-Content -Path "d:\newf.txt"

# converting the format string to a pipeline function 
filter fmt { "c{0:d2}d" -f $_ }
8..12 | fmt | Set-Content -Path "d:\newf.txt"

# filter can be passed positional parameters in addition to the pipeline element
filter Map-ToUrl { $args[0] -f $_ }

1..80 | Map-ToUrl 'https://example.com/images/abc-{0:d}.jpg'
      | % { Invoke-WebRequest -HttpVersion 2.0 -Uri $_ -OutFile ("C:\temp\files\" + $_.split("/")[-1]);
            echo $_; }

function Get-Latest {
    param (
        [string]$Spec = "qqq*.csv"
    )
    Get-ChildItem -Path "~\Downloads" -Filter $Spec |
    Sort-Object -Property LastWriteTime -Descending -Top 1
}


# stateful pipeline function sums the length property on pipeline objects
function flen { begin { $sum = 0 }; process { $sum = $sum + $_.Length }; end { $sum } }

dir -File | flen

# can also do this using builtin
dir -File | Measure-Object Length -Sum

# join-path will create a new path from parts and resolve wildcards
$p = Join-Path -Path 'd:\test\*' -Childpath "other" -Resolve

# build path, find 1 file and print content
Join-Path d:\test\* -ChildPath other -Resolve |
    Get-ChildItem -Filter *.txt |
    Sort-Object -Property Length -Descending -Top 1 |
    Get-Content

"asdf house 54 ant house 12 asd" | Select-String "house (?<v>\w+)" -AllMatches | % { $_.Matches[0].Groups['v'].Value }
$x = "asdf house 54 ant house 12 asd" | Select-String "house (?<v>\w+)" -AllMatches
foreach ($m in $x.Matches) {
    foreach ($g in $m.Groups) {
        '{0} = {1}' -f $g.Name, $g.Value
    }
}

$a = 'placesforpeople', 'robinhoodenergy', 'greennetwork'
$a | Select-String ('rhe'.ToCharArray() -join '.*')
dir | ? { $_.Name -match 'e.*s' }