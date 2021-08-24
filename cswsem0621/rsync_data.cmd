gsname='gs0823'
echo "RSYNC GRIDSEARCH DATA FROM:"
echo ${gsname}
echo
rsync -r -vam --progress abeukers@scotty.princeton.edu:/jukebox/norman/abeukers/sem/cswsem/CSW/cswsem0621/data/${gsname}/* /Users/abeukers/wd/csw/cswsem0621/data/${gsname}