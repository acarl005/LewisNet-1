# many of the tiles contain nothing but white. those should be removed
# a 150x150 PNG of all white pixels happens to be 425 bytes
# parse ls command and filter those out and delete them
cd tiles
ls -l | grep 425 | awk '{print $9}' | xargs rm

