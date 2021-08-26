gsname='gs0825'
echo "RSYNC GRIDSEARCH DATA FROM:"
echo ${gsname}
echo
# rsync -r -vam --progress abeukers@tiger.princeton.edu:/home/abeukers/wd/CSW/cswsem0621/data/${gsname}/* /Users/abeukers/wd/csw/cswsem0621/data/${gsname}
rsync -r -vam --progress abeukers@scotty.princeton.edu:/jukebox/norman/abeukers/sem/cswsem/CSW/cswsem0621/data/${gsname}/* /Users/abeukers/wd/csw/cswsem0621/data/${gsname}